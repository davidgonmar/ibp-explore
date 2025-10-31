import copy
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import prune
from torch.ao.pruning import WeightNormSparsifier
from torchao.sparsity import WandaSparsifier

from sparse.taylor_prune import prune_taylor_unstructured
from models import MLP, resnet20
from utils import evaluate_with_classifier
from utils import (
    DEVICE,
    BATCH_SIZE,
    get_test_and_mi_loaders,
    estimate_IZY_and_IXZ,
    save_results_json,
)


def _get_prunable_modules(model):
    mods = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            mods.append((name, mod))
    return mods


def _ao_sparsify_model_with_config(model, sparsifier):
    sparse_config = []
    for fqn, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            sparse_config.append({"tensor_fqn": f"{fqn}.weight"})
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()
    sparsifier.squash_mask()


def _wanda_sparsify_model_with_config(model, amount, calib_subset):
    sparsifier = WandaSparsifier(sparsity_level=amount)
    sparse_config = []
    for fqn, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            sparse_config.append({"tensor_fqn": f"{fqn}.weight"})
    sparsifier.prepare(model, sparse_config)
    calib_loader = DataLoader(
        calib_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    for x, _ in calib_loader:
        x = x.to(DEVICE)
        model(x)
    sparsifier.step()
    sparsifier.squash_mask()


def _taylor_unstructured(model, amount, calib_subset):
    ce_loss = torch.nn.CrossEntropyLoss()
    calib_loader = DataLoader(
        calib_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    prune_taylor_unstructured(
        model=model,
        dataloader=calib_loader,
        loss_fn=ce_loss,
        sparsity=amount,
        device=DEVICE,
        num_batches=10,
        forward_fn=None,
        skip_modules=(),
    )


def _effective_weight_and_bias(mod):
    if hasattr(mod, "weight_mask") and hasattr(mod, "weight_orig"):
        w_eff = mod.weight_orig * mod.weight_mask
    else:
        w_eff = mod.weight
    b = getattr(mod, "bias", None)
    return w_eff, b


def _baseline_bits(model):
    total_params = 0
    for _, m in _get_prunable_modules(model):
        total_params += m.weight.numel()
        if m.bias is not None:
            total_params += m.bias.numel()
    return total_params * 32


def _compression_ratio_pruned(model, baseline_bits):
    nonzero_params = 0
    for _, m in _get_prunable_modules(model):
        w_eff, b = _effective_weight_and_bias(m)
        nonzero_params += (w_eff != 0).sum().item()
        if b is not None:
            nonzero_params += b.numel()
    compressed_bits = nonzero_params * 32
    return compressed_bits / baseline_bits


def _build_data_and_model(model_name):
    if model_name == "MLP":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_ds = datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transform,
        )
        model = MLP().to(DEVICE)
        ckpt_path = "mlp_mnist.pth"
    elif model_name == "ResNet20":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train_ds = datasets.CIFAR10(
            "./data",
            train=True,
            download=True,
            transform=transform,
        )
        model = resnet20().to(DEVICE)
        ckpt_path = "resnet20_cifar.pth"
    else:
        raise ValueError("model_name must be 'MLP' or 'ResNet20'")
    return train_ds, model, ckpt_path


def main(model_name, output_path):
    dataset_name = "mnist" if model_name == "MLP" else "cifar10"
    train_ds, model_base, ckpt_path = _build_data_and_model(model_name)
    test_loader, test_loader_mi = get_test_and_mi_loaders(dataset_name)
    model_base.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    baseline_bits = _baseline_bits(model_base)
    ratios = (
        [
            0.0,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            0.975,
            0.99,
        ]
        if model_name == "MLP"
        else [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.99,
        ]
    )

    PRUNING_STRATEGIES = {
        "global_l1": lambda model, params, amount: prune.global_unstructured(
            params,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        ),
        "global_random": lambda model, params, amount: prune.global_unstructured(
            params,
            pruning_method=prune.RandomUnstructured,
            amount=amount,
        ),
        "layerwise_l1": lambda model, params, amount: [
            prune.l1_unstructured(m, name, amount=amount) for (m, name) in params
        ],
        "layerwise_random": lambda model, params, amount: [
            prune.random_unstructured(m, name, amount=amount) for (m, name) in params
        ],
        "structured_l2_channels": lambda model, params, amount: [
            prune.ln_structured(m, name, amount=amount, n=2, dim=0)
            for (m, name) in params
        ],
        "semi_structured_nm_weightnorm": lambda model, params, amount: _ao_sparsify_model_with_config(
            model,
            WeightNormSparsifier(
                sparsity_level=1.0,
                sparse_block_shape=(1, 4),
                zeros_per_block=int(max(0, min(4, round(amount * 4)))),
            ),
        ),
        "wanda_unstructured": lambda model, params, amount, calib_subset=None: _wanda_sparsify_model_with_config(
            model,
            amount,
            calib_subset,
        ),
        "taylor_unstructured": lambda model, params, amount, calib_subset=None: _taylor_unstructured(
            model,
            amount,
            calib_subset,
        ),
    }

    PRUNING_STRATEGIES = (
        PRUNING_STRATEGIES
        if model_name == "MLP"
        else {
            k: v
            for k, v in PRUNING_STRATEGIES.items()
            if k
            in {"global_l1", "global_random", "layerwise_l1", "taylor_unstructured"}
        }
    )
    all_results = {}
    for strategy_name, strategy_fn in PRUNING_STRATEGIES.items():
        accs_by_decoder = {}
        izy_list = []
        ixz_list = []
        compr_list = []
        print(f"\n=== Pruning strategy: {strategy_name} ===")
        for ratio in ratios:
            model = copy.deepcopy(model_base)
            prunable_modules = _get_prunable_modules(model)
            parameters_to_prune = [(m, "weight") for (_, m) in prunable_modules]
            if ratio > 0.0:
                if "wanda" in strategy_name or "taylor" in strategy_name:
                    strategy_fn(
                        model,
                        parameters_to_prune,
                        ratio,
                        calib_subset=train_ds,
                    )
                else:
                    strategy_fn(model, parameters_to_prune, ratio)
            compr_val = _compression_ratio_pruned(model, baseline_bits)
            eval_results = evaluate_with_classifier(
                model,
                test_loader,
                classifiers=["linear", "mlp"],
                num_classes=10,
            )
            for dec_name, info in eval_results.items():
                if dec_name not in accs_by_decoder:
                    accs_by_decoder[dec_name] = []
                accs_by_decoder[dec_name].append(info["accuracy"])
            I_ZY, I_XZ = estimate_IZY_and_IXZ(model, dataset_name, test_loader_mi)
            izy_list.append(I_ZY)
            ixz_list.append(I_XZ)
            compr_list.append(compr_val)
            acc_print_parts = []
            for dec_name, info in eval_results.items():
                acc_print_parts.append(
                    f"{dec_name} Acc: {info['accuracy'] * 100:6.2f}%"
                )
            acc_print = " | ".join(acc_print_parts)
            print(
                f"Prune ratio: {ratio:.2f} | "
                f"{acc_print} | "
                f"I(Z;Y): {I_ZY:7.3f} bits | I(X;Z): {I_XZ:7.3f} bits"
            )
        all_results[strategy_name] = {
            "ratios": ratios,
            "accs": accs_by_decoder,
            "izy": izy_list,
            "ixz": ixz_list,
            "compr": compr_list,
        }
    save_results_json(all_results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="MLP",
        choices=["MLP", "ResNet20"],
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/pruning_all_layers.json",
    )
    args = parser.parse_args()
    main(args.model_name, args.output_path)
