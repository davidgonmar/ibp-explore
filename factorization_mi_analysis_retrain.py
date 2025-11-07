import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from balf.utils import (
    cifar10_mean,
    cifar10_std,
    seed_everything,
    get_all_convs_and_linears,
    make_factorization_cache_location,
)
from balf.factorization.factorize import (
    to_low_rank_activation_aware_auto,
    to_low_rank_activation_aware_manual,
    to_low_rank_manual,
    collect_activation_cache,
)
from utils import (
    DEVICE,
    get_test_and_mi_loaders,
    evaluate_with_classifier,
    estimate_IZY_and_IXZ,
    save_results_json,
    accuracy_class_differentiate_with_classifiers,
)
from models import resnet20


def _count_params(m):
    total = 0
    for p in m.parameters():
        total += p.numel()
    return total


def _factorize_model(
    base_model, method, ratio, activation_cache, layer_keys, cache_dir
):
    if method == "balf":
        model_lr = to_low_rank_activation_aware_auto(
            base_model,
            activation_cache,
            keys=layer_keys,
            ratio_to_keep=ratio,
            metric="params",
            inplace=False,
            save_dir=cache_dir,
        )
    elif method == "aa_rank":
        cfg_dict = {
            kk: {"name": "rank_ratio_to_keep", "value": ratio} for kk in layer_keys
        }
        model_lr = to_low_rank_activation_aware_manual(
            base_model,
            activation_cache,
            cfg_dict=cfg_dict,
            inplace=False,
            save_dir=cache_dir,
        )
    elif method == "plain_rank":
        cfg_dict = {
            kk: {"name": "rank_ratio_to_keep", "value": ratio} for kk in layer_keys
        }
        model_lr = to_low_rank_manual(
            base_model,
            cfg_dict=cfg_dict,
            inplace=False,
        )
    else:
        raise ValueError("invalid method")
    return model_lr


def _retrain_model(model, train_loader, num_epochs, lr):
    model = model.to(DEVICE)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet20"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10"])
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        required=True,
        choices=["balf", "aa_rank", "plain_rank"],
    )
    parser.add_argument("--ratios", type=float, nargs="+", required=True)
    parser.add_argument("--calib_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrain_epochs", type=int, default=5)
    parser.add_argument("--retrain_lr", type=float, default=1e-3)
    parser.add_argument("--retrain_batch_size", type=int, default=128)
    parser.add_argument("--retrain_size", type=int, default=5000)
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.model == "resnet20":
        base_model = resnet20().to(DEVICE)
    else:
        raise ValueError("unsupported model")

    state = torch.load(args.checkpoint, map_location=DEVICE)
    base_model.load_state_dict(state)
    base_model.eval()

    layer_keys = list(get_all_convs_and_linears(base_model))

    if args.dataset == "cifar10":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
        )
        train_ds = datasets.CIFAR10(
            root="data", train=True, transform=transform, download=True
        )
        g = torch.Generator()
        g.manual_seed(args.seed)
        idx = torch.randperm(len(train_ds), generator=g)[: args.calib_size]
        subset = torch.utils.data.Subset(train_ds, idx)
        calib_loader = DataLoader(subset, batch_size=256)

        g_retrain = torch.Generator()
        g_retrain.manual_seed(args.seed + 1)
        idx_retrain = torch.randperm(len(train_ds), generator=g_retrain)[
            : args.retrain_size
        ]
        subset_retrain = torch.utils.data.Subset(train_ds, idx_retrain)
        retrain_loader = DataLoader(
            subset_retrain, batch_size=args.retrain_batch_size, shuffle=True
        )
    else:
        raise ValueError("unsupported dataset")

    activation_cache = collect_activation_cache(
        base_model, calib_loader, keys=layer_keys
    )

    test_loader, mi_loader = get_test_and_mi_loaders(args.dataset)

    params_orig = _count_params(base_model)

    cache_dir_prefix = args.model + "_" + args.checkpoint.split("/")[-1].split(".")[0]

    eval_base = evaluate_with_classifier(
        base_model,
        test_loader,
        classifiers=["linear", "mlp"],
        num_classes=10,
    )
    (I_ZY_base, per_class_binary_info_base), I_XZ_base = estimate_IZY_and_IXZ(
        base_model,
        args.dataset,
        mi_loader,
    )

    results = {}

    for method in args.methods:
        cache_dir = make_factorization_cache_location(
            cache_dir_prefix,
            args.calib_size,
            args.dataset,
            "factorize_sweep",
            args.seed,
        )

        labels = ["original"]
        izy_list = [I_ZY_base]
        per_class_binary_info_list = [per_class_binary_info_base]
        ixz_list = [I_XZ_base]
        compr_list = [1.0]
        accs_per_class_differentiate_by_decoder = (
            accuracy_class_differentiate_with_classifiers(
                base_model, test_loader, classifiers=["linear", "mlp"], num_classes=10
            )
        )
        for dec_name in accs_per_class_differentiate_by_decoder.keys():
            accs_per_class_differentiate_by_decoder[dec_name] = [
                accs_per_class_differentiate_by_decoder[dec_name]
            ]
        accs_by_decoder = {}
        for dec_name, info in eval_base.items():
            accs_by_decoder[dec_name] = [info["accuracy"]]

        parts = []
        for dec_name, info in eval_base.items():
            parts.append(f"{dec_name} Acc: {info['accuracy'] * 100:6.2f}%")
        acc_print = " | ".join(parts)
        print(
            f"[method={method} | original] {acc_print} | I(Z;Y): {I_ZY_base:7.3f} bits | I(X;Z): {I_XZ_base:7.3f} bits | compr=1.0000"
        )

        for ratio in args.ratios:
            model_lr = _factorize_model(
                base_model,
                method,
                ratio,
                activation_cache,
                layer_keys,
                cache_dir,
            )
            model_lr = model_lr.to(DEVICE)
            model_lr.eval()

            model_lr = _retrain_model(
                model_lr, retrain_loader, args.retrain_epochs, args.retrain_lr
            )

            params_lr = _count_params(model_lr)
            compr_val = params_lr / params_orig

            eval_lr = evaluate_with_classifier(
                model_lr,
                test_loader,
                classifiers=["linear", "mlp"],
                num_classes=10,
            )
            (I_ZY, per_class_binary_info_lr), I_XZ = estimate_IZY_and_IXZ(
                model_lr,
                args.dataset,
                mi_loader,
            )

            labels.append(f"{ratio:.6f}")
            izy_list.append(I_ZY)
            ixz_list.append(I_XZ)
            compr_list.append(compr_val)
            per_class_binary_info_list.append(per_class_binary_info_lr)
            for dec_name, info in eval_lr.items():
                if dec_name not in accs_by_decoder:
                    accs_by_decoder[dec_name] = []
                accs_by_decoder[dec_name].append(info["accuracy"])

            parts = []
            for dec_name, info in eval_lr.items():
                parts.append(f"{dec_name} Acc: {info['accuracy'] * 100:6.2f}%")
            acc_print = " | ".join(parts)
            print(
                f"[method={method} | ratio={ratio:.6f}] {acc_print} | I(Z;Y): {I_ZY:7.3f} bits | I(X;Z): {I_XZ:7.3f} bits | compr={compr_val:.4f}"
            )

            accs_per_class_differentiate_lr = (
                accuracy_class_differentiate_with_classifiers(
                    model_lr, test_loader, classifiers=["linear", "mlp"], num_classes=10
                )
            )

            for dec_name, per_class_errors in accs_per_class_differentiate_lr.items():
                if dec_name not in accs_per_class_differentiate_by_decoder:
                    accs_per_class_differentiate_by_decoder[dec_name] = []
                accs_per_class_differentiate_by_decoder[dec_name].append(
                    per_class_errors
                )

        results[method] = {
            "labels": labels,
            "accs": accs_by_decoder,
            "izy": izy_list,
            "per_class_binary_info": per_class_binary_info_list,
            "ixz": ixz_list,
            "compr": compr_list,
            "accs_per_class_differentiate": accs_per_class_differentiate_by_decoder,
        }

    final_out = {"methods": results}

    save_results_json(final_out, args.out_json)


if __name__ == "__main__":
    main()
