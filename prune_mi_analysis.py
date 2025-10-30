import copy
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import prune
from torch.ao.pruning import WeightNormSparsifier
from torchao.sparsity import WandaSparsifier

from sparse.taylor_prune import prune_taylor_unstructured
from models import MLP
from utils import evaluate_with_classifier
from utils import (
    DEVICE,
    BATCH_SIZE,
    get_test_and_mi_loaders,
    estimate_IZY_and_IXZ,
    save_results_json,
)


def _ao_sparsify_model_with_config(model, sparsifier):
    sparse_config = []
    for fqn, mod in model.named_modules():
        if fqn in ("fc1", "fc2") and isinstance(mod, torch.nn.Linear):
            sparse_config.append({"tensor_fqn": f"{fqn}.weight"})
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()
    sparsifier.squash_mask()


def _wanda_sparsify_model_with_config(model, amount, calib_subset):
    sparsifier = WandaSparsifier(sparsity_level=amount)
    sparse_config = []
    for fqn, mod in model.named_modules():
        if fqn in ("fc1", "fc2") and isinstance(mod, torch.nn.Linear):
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
        prune.ln_structured(m, name, amount=amount, n=2, dim=0) for (m, name) in params
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


def _linear_effective_weight_and_bias(mod):
    if hasattr(mod, "weight_mask") and hasattr(mod, "weight_orig"):
        w_eff = mod.weight_orig * mod.weight_mask
    else:
        w_eff = mod.weight
    b = mod.bias
    return w_eff, b


def _baseline_bits(model):
    total_params = 0
    for m in (model.fc1, model.fc2, model.fc3):
        total_params += m.weight.numel() + m.bias.numel()
    return total_params * 32


def _compression_ratio_pruned(model, baseline_bits):
    nonzero_params = 0
    for m in (model.fc1, model.fc2, model.fc3):
        w_eff, b = _linear_effective_weight_and_bias(m)
        nonzero_params += (w_eff != 0).sum().item()
        nonzero_params += b.numel()
    compressed_bits = nonzero_params * 32
    return compressed_bits / baseline_bits


def main():
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_loader, test_loader_mi = get_test_and_mi_loaders()

    model_base = MLP().to(DEVICE)
    model_base.load_state_dict(torch.load("mlp_mnist.pth", map_location=DEVICE))

    baseline_bits = _baseline_bits(model_base)

    ratios = [
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

    all_results = {}

    for strategy_name, strategy_fn in PRUNING_STRATEGIES.items():
        accs = []
        izy_list = []
        ixz_list = []
        compr_list = []
        err_counts_list = []
        err_shares_list = []
        totals_per_class = None

        print(f"\n=== Pruning strategy: {strategy_name} ===")
        for ratio in ratios:
            model = copy.deepcopy(model_base)

            parameters_to_prune = [
                (model.fc1, "weight"),
                (model.fc2, "weight"),
                (model.fc3, "weight"),
            ]

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

            eval_info, clf = evaluate_with_classifier(
                model,
                test_loader,
                classifier="mlp",
                num_classes=10,
            )
            acc = eval_info["accuracy"]

            breakdown = {
                "errors_per_class": eval_info["errors_per_class"],
                "total_per_class": eval_info["total_per_class"],
                "total_errors": eval_info["total_errors"],
                "error_share": eval_info["error_share"],
            }

            I_ZY, I_XZ = estimate_IZY_and_IXZ(model, test_loader_mi)

            if totals_per_class is None:
                totals_per_class = breakdown["total_per_class"]

            accs.append(acc)
            izy_list.append(I_ZY)
            ixz_list.append(I_XZ)
            compr_list.append(compr_val)
            err_counts_list.append(breakdown["errors_per_class"])
            err_shares_list.append(breakdown["error_share"])

            shares_pct = [f"{100 * s:.1f}%" for s in breakdown["error_share"]]
            shares_str = ", ".join(
                [f"class {i}: {p}" for i, p in enumerate(shares_pct)]
            )
            print(
                f"Prune ratio: {ratio:.2f} | "
                f"Probe Acc: {acc * 100:6.2f}% | "
                f"I(Z;Y): {I_ZY:7.3f} bits | I(X;Z): {I_XZ:7.3f} bits"
            )
            print(
                " Error share by true class (percent of all mistakes): " f"{shares_str}"
            )

        all_results[strategy_name] = {
            "ratios": ratios,
            "accs": accs,
            "izy": izy_list,
            "ixz": ixz_list,
            "compr": compr_list,
            "err_counts": err_counts_list,
            "err_shares": err_shares_list,
            "totals": totals_per_class,
        }

    save_results_json(all_results, "results/pruning_all_layers.json")


if __name__ == "__main__":
    main()
