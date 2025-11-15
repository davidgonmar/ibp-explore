import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import prune
from sparse.taylor_prune import prune_taylor_unstructured
from balf.utils import (
    cifar10_mean,
    cifar10_std,
    seed_everything,
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


def _get_prunable_modules(model):
    mods = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            mods.append((name, mod))
    return mods


def _baseline_bits(model):
    total_params = 0
    for _, m in _get_prunable_modules(model):
        total_params += m.weight.numel()
        if m.bias is not None:
            total_params += m.bias.numel()
    return total_params * 32


def _effective_weight_and_bias(mod):
    if hasattr(mod, "weight_mask") and hasattr(mod, "weight_orig"):
        w_eff = mod.weight_orig * mod.weight_mask
    else:
        w_eff = mod.weight
    b = getattr(mod, "bias", None)
    return w_eff, b


def _compression_ratio_pruned(model, baseline_bits):
    nonzero_params = 0
    for _, m in _get_prunable_modules(model):
        w_eff, b = _effective_weight_and_bias(m)
        nonzero_params += (w_eff != 0).sum().item()
        if b is not None:
            nonzero_params += b.numel()
    compressed_bits = nonzero_params * 32
    return compressed_bits / baseline_bits


def _prune_model(base_model, method, ratio, calib_loader):
    model = copy.deepcopy(base_model)
    prunable_modules = _get_prunable_modules(model)
    parameters_to_prune = [(m, "weight") for (_, m) in prunable_modules]

    if ratio > 0.0:
        if method == "global_l1":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=ratio,
            )
        elif method == "global_random":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=ratio,
            )
        elif method == "layerwise_l1":
            for m, name in parameters_to_prune:
                prune.l1_unstructured(m, name, amount=ratio)
        elif method == "taylor_unstructured":
            ce_loss = torch.nn.CrossEntropyLoss()
            prune_taylor_unstructured(
                model=model,
                dataloader=calib_loader,
                loss_fn=ce_loss,
                sparsity=ratio,
                device=DEVICE,
                num_batches=10,
                forward_fn=None,
                skip_modules=(),
            )
        else:
            raise ValueError("invalid method")

    for m, name in parameters_to_prune:
        prune.remove(m, name)

    return model


NOISE_MODE = "fancy"


def _retrain_model(model, train_loader, num_epochs, lr, label_noise):
    model = model.to(DEVICE)
    model.train()

    if label_noise > 0.0:
        base_ds = train_loader.dataset

        if not hasattr(base_ds, "_noisy_labels"):
            if NOISE_MODE == "fancy":
                ys = []
                for i in range(len(base_ds)):
                    _, yi = base_ds[i]
                    ys.append(yi)
                clean = torch.tensor(ys, dtype=torch.long)
                K = int(clean.max().item()) + 1
                identity = torch.eye(K, dtype=torch.float32)
                T = (1.0 - label_noise) * identity + (label_noise / (K - 1)) * (
                    torch.ones((K, K), dtype=torch.float32) - identity
                )
                P = T[clean]
                noisy = torch.multinomial(P, 1, replacement=True).squeeze(1)
                base_ds._noisy_labels = noisy
            else:
                import random

                ys = []
                for i in range(len(base_ds)):
                    _, yi = base_ds[i]
                    ys.append(yi)
                noisy = []
                for yi in ys:
                    if random.random() < label_noise:
                        noisy.append(random.randint(0, 9))
                    else:
                        noisy.append(yi)
                base_ds._noisy_labels = noisy

        class _NoisyWrapper(torch.utils.data.Dataset):
            def __init__(self, base, noisy):
                self.base = base
                self.noisy = noisy

            def __len__(self):
                return len(self.base)

            def __getitem__(self, idx):
                x, _ = self.base[idx]
                return x, int(self.noisy[idx])

        train_loader = DataLoader(
            _NoisyWrapper(base_ds, base_ds._noisy_labels),
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=getattr(train_loader, "num_workers", 0),
            pin_memory=getattr(train_loader, "pin_memory", False),
            drop_last=getattr(train_loader, "drop_last", False),
        )

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

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
        choices=["global_l1", "global_random", "layerwise_l1", "taylor_unstructured"],
    )
    parser.add_argument("--ratios", type=float, nargs="+", required=True)
    parser.add_argument("--calib_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrain_epochs", type=int, default=5)
    parser.add_argument("--retrain_lr", type=float, default=1e-3)
    parser.add_argument("--retrain_batch_size", type=int, default=128)
    parser.add_argument("--retrain_size", type=int, default=5000)
    parser.add_argument("--retrain_label_noise", type=float, default=0.0)
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.model == "resnet20":
        base_model = resnet20().to(DEVICE)
    else:
        raise ValueError("unsupported model")

    state = torch.load(args.checkpoint, map_location=DEVICE)
    base_model.load_state_dict(state)
    base_model.eval()

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

    test_loader, mi_loader = get_test_and_mi_loaders(args.dataset)

    baseline_bits = _baseline_bits(base_model)

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

        acc_vs_compr_curves = {}
        for classifier_name in eval_base.keys():
            acc_vs_compr_curves[classifier_name] = {"before": [], "after": []}
        retrain_ratios = []

        labels_before = ["original"]
        izy_list_before = [I_ZY_base]
        per_class_binary_info_list_before = [per_class_binary_info_base]
        ixz_list_before = [I_XZ_base]
        compr_list_before = [1.0]
        accs_by_decoder_before = {}
        for dec_name, info in eval_base.items():
            accs_by_decoder_before[dec_name] = [info["accuracy"]]
            acc_vs_compr_curves[dec_name]["before"].append(info["accuracy"])
        accs_per_class_differentiate_by_decoder_before = {}
        for dec_name in accs_per_class_differentiate_by_decoder.keys():
            accs_per_class_differentiate_by_decoder_before[dec_name] = [
                accs_per_class_differentiate_by_decoder[dec_name]
            ]

        labels_after = ["original"]
        izy_list_after = [I_ZY_base]
        per_class_binary_info_list_after = [per_class_binary_info_base]
        ixz_list_after = [I_XZ_base]
        compr_list_after = [1.0]
        accs_by_decoder_after = {}
        for dec_name, info in eval_base.items():
            accs_by_decoder_after[dec_name] = [info["accuracy"]]
            acc_vs_compr_curves[dec_name]["after"].append(info["accuracy"])
        accs_per_class_differentiate_by_decoder_after = {}
        for dec_name in accs_per_class_differentiate_by_decoder.keys():
            accs_per_class_differentiate_by_decoder_after[dec_name] = [
                accs_per_class_differentiate_by_decoder[dec_name]
            ]

        parts = []
        for dec_name, info in eval_base.items():
            parts.append(f"{dec_name} Acc: {info['accuracy'] * 100:6.2f}%")
        acc_print = " | ".join(parts)
        print(
            f"[method={method} | original] {acc_print} | I(Z;Y): {I_ZY_base:7.3f} bits | I(X;Z): {I_XZ_base:7.3f} bits | compr=1.0000"
        )

        for ratio in args.ratios:
            model_pruned = _prune_model(
                base_model,
                method,
                ratio,
                calib_loader,
            )
            model_pruned = model_pruned.to(DEVICE)
            model_pruned.eval()

            compr_val = _compression_ratio_pruned(model_pruned, baseline_bits)

            eval_pruned_before = evaluate_with_classifier(
                model_pruned,
                test_loader,
                classifiers=["linear", "mlp"],
                num_classes=10,
            )
            (I_ZY_before, per_class_binary_info_pruned_before), I_XZ_before = (
                estimate_IZY_and_IXZ(
                    model_pruned,
                    args.dataset,
                    mi_loader,
                )
            )

            accs_per_class_differentiate_pruned_before = (
                accuracy_class_differentiate_with_classifiers(
                    model_pruned,
                    test_loader,
                    classifiers=["linear", "mlp"],
                    num_classes=10,
                )
            )

            labels_before.append(f"{ratio:.6f}")
            izy_list_before.append(I_ZY_before)
            ixz_list_before.append(I_XZ_before)
            compr_list_before.append(compr_val)
            per_class_binary_info_list_before.append(
                per_class_binary_info_pruned_before
            )
            for dec_name, info in eval_pruned_before.items():
                if dec_name not in accs_by_decoder_before:
                    accs_by_decoder_before[dec_name] = []
                accs_by_decoder_before[dec_name].append(info["accuracy"])
                acc_vs_compr_curves[dec_name]["before"].append(info["accuracy"])

            for (
                dec_name,
                per_class_errors,
            ) in accs_per_class_differentiate_pruned_before.items():
                if dec_name not in accs_per_class_differentiate_by_decoder_before:
                    accs_per_class_differentiate_by_decoder_before[dec_name] = []
                accs_per_class_differentiate_by_decoder_before[dec_name].append(
                    per_class_errors
                )

            parts_before = []
            for dec_name, info in eval_pruned_before.items():
                parts_before.append(f"{dec_name} Acc: {info['accuracy'] * 100:6.2f}%")
            acc_print_before = " | ".join(parts_before)
            print(
                f"[method={method} | ratio={ratio:.6f} | BEFORE retrain] {acc_print_before} | I(Z;Y): {I_ZY_before:7.3f} bits | I(X;Z): {I_XZ_before:7.3f} bits | compr={compr_val:.4f}"
            )

            model_pruned = _retrain_model(
                model_pruned,
                retrain_loader,
                args.retrain_epochs,
                args.retrain_lr,
                args.retrain_label_noise,
            )

            eval_pruned_after = evaluate_with_classifier(
                model_pruned,
                test_loader,
                classifiers=["linear", "mlp"],
                num_classes=10,
            )
            (I_ZY_after, per_class_binary_info_pruned_after), I_XZ_after = (
                estimate_IZY_and_IXZ(
                    model_pruned,
                    args.dataset,
                    mi_loader,
                )
            )

            labels_after.append(f"{ratio:.6f}")
            izy_list_after.append(I_ZY_after)
            ixz_list_after.append(I_XZ_after)
            compr_list_after.append(compr_val)
            per_class_binary_info_list_after.append(per_class_binary_info_pruned_after)
            for dec_name, info in eval_pruned_after.items():
                if dec_name not in accs_by_decoder_after:
                    accs_by_decoder_after[dec_name] = []
                accs_by_decoder_after[dec_name].append(info["accuracy"])
                acc_vs_compr_curves[dec_name]["after"].append(info["accuracy"])

            parts_after = []
            for dec_name, info in eval_pruned_after.items():
                parts_after.append(f"{dec_name} Acc: {info['accuracy'] * 100:6.2f}%")
            acc_print_after = " | ".join(parts_after)
            print(
                f"[method={method} | ratio={ratio:.6f} | AFTER retrain] {acc_print_after} | I(Z;Y): {I_ZY_after:7.3f} bits | I(X;Z): {I_XZ_after:7.3f} bits | compr={compr_val:.4f}"
            )

            accs_per_class_differentiate_pruned_after = (
                accuracy_class_differentiate_with_classifiers(
                    model_pruned,
                    test_loader,
                    classifiers=["linear", "mlp"],
                    num_classes=10,
                )
            )

            for (
                dec_name,
                per_class_errors,
            ) in accs_per_class_differentiate_pruned_after.items():
                if dec_name not in accs_per_class_differentiate_by_decoder_after:
                    accs_per_class_differentiate_by_decoder_after[dec_name] = []
                accs_per_class_differentiate_by_decoder_after[dec_name].append(
                    per_class_errors
                )

            labels.append(f"{ratio:.6f}")
            izy_list.append(I_ZY_after)
            ixz_list.append(I_XZ_after)
            compr_list.append(compr_val)
            per_class_binary_info_list.append(per_class_binary_info_pruned_after)
            for dec_name, info in eval_pruned_after.items():
                if dec_name not in accs_by_decoder:
                    accs_by_decoder[dec_name] = []
                accs_by_decoder[dec_name].append(info["accuracy"])

            for (
                dec_name,
                per_class_errors,
            ) in accs_per_class_differentiate_pruned_after.items():
                if dec_name not in accs_per_class_differentiate_by_decoder:
                    accs_per_class_differentiate_by_decoder[dec_name] = []
                accs_per_class_differentiate_by_decoder[dec_name].append(
                    per_class_errors
                )

            retrain_ratios.append(compr_val)

        results[method] = {
            "labels": labels,
            "accs": accs_by_decoder,
            "izy": izy_list,
            "per_class_binary_info": per_class_binary_info_list,
            "ixz": ixz_list,
            "compr": compr_list,
            "accs_per_class_differentiate": accs_per_class_differentiate_by_decoder,
            "before_retrain": {
                "labels": labels_before,
                "accs": accs_by_decoder_before,
                "izy": izy_list_before,
                "per_class_binary_info": per_class_binary_info_list_before,
                "ixz": ixz_list_before,
                "compr": compr_list_before,
                "accs_per_class_differentiate": accs_per_class_differentiate_by_decoder_before,
            },
            "after_retrain": {
                "labels": labels_after,
                "accs": accs_by_decoder_after,
                "izy": izy_list_after,
                "per_class_binary_info": per_class_binary_info_list_after,
                "ixz": ixz_list_after,
                "compr": compr_list_after,
                "accs_per_class_differentiate": accs_per_class_differentiate_by_decoder_after,
            },
            "acc_vs_compr_curves": acc_vs_compr_curves,
        }

    final_out = {"methods": results}

    save_results_json(final_out, args.out_json)


if __name__ == "__main__":
    main()
