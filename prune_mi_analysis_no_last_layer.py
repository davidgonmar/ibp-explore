import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import prune
from omegaconf import DictConfig
from mutual_info.estimate_mi import estimate_mi_zy, estimate_mi_zx
from models import MLP
import matplotlib.pyplot as plt
import os
from torch.ao.pruning import WeightNormSparsifier
from torchao.sparsity import WandaSparsifier
import copy
import numpy as np
import math
from sparse.taylor_prune import prune_taylor_unstructured

BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MI_SAMPLE = 4096

cfg = DictConfig({"dataset": {"name": "mnist", "size": 28, "num_channels": 1}})


def _binary_entropy_bits(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _fano_error_upper_bound(
    I_bits: float, K: int = 10, H_Y_bits: float = None, tol: float = 1e-7
) -> float:
    if H_Y_bits is None:
        H_Y_bits = math.log2(K)
    target = H_Y_bits - I_bits
    if target <= 0.0:
        return 0.0
    lo, hi = 0.0, 1.0 - 1.0 / K

    def g(p):
        return _binary_entropy_bits(p) + p * math.log2(K - 1) - target

    if g(hi) < 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = g(mid)
        if val > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def fano_upper_accuracy_from_I(
    I_value: float, K: int = 10, H_Y_bits: float = None
) -> float:
    I_bits = I_value
    Pe = _fano_error_upper_bound(I_bits, K=K, H_Y_bits=H_Y_bits)
    return 1.0 - Pe


def _ao_sparsify_model_with_config(model, sparsifier):
    sparse_config = []
    for fqn, mod in model.named_modules():
        if fqn in ("fc1", "fc2") and isinstance(mod, torch.nn.Linear):
            sparse_config.append({"tensor_fqn": f"{fqn}.weight"})
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()
    sparsifier.squash_mask()


mnist_train = datasets.MNIST(
    "./data", train=True, download=True, transform=transforms.ToTensor()
)
mnist_subset = torch.utils.data.Subset(
    mnist_train, torch.randperm(len(mnist_train))[:MI_SAMPLE]
)


def _wanda_sparsify(model, amount):
    sparse_config = []
    for fqn, mod in model.named_modules():
        if fqn in ("fc1", "fc2") and isinstance(mod, torch.nn.Linear):
            sparse_config.append({"tensor_fqn": f"{fqn}.weight"})
    sparsifier = WandaSparsifier(
        sparsity_level=amount,
    )
    sparsifier.prepare(model, sparse_config)
    loader = DataLoader(mnist_subset, batch_size=BATCH_SIZE, shuffle=False)
    for x, _ in loader:
        x = x.to(DEVICE)
        model(x)
    sparsifier.step()
    sparsifier.squash_mask()


_ce_loss = torch.nn.CrossEntropyLoss()
_taylor_loader = DataLoader(mnist_subset, batch_size=BATCH_SIZE, shuffle=False)


def _taylor_unstructured_no_bias(model, _params_ignored, amount: float):
    if amount <= 0.0:
        return
    prune_taylor_unstructured(
        model=model,
        dataloader=_taylor_loader,
        loss_fn=_ce_loss,
        sparsity=amount,
        device=DEVICE,
        num_batches=10,
        forward_fn=None,
        skip_modules=(),
    )


PRUNING_STRATEGIES = {
    "global_l1": lambda model, params, amount: prune.global_unstructured(
        params, pruning_method=prune.L1Unstructured, amount=amount
    ),
    "global_random": lambda model, params, amount: prune.global_unstructured(
        params, pruning_method=prune.RandomUnstructured, amount=amount
    ),
    "layerwise_l1": lambda model, params, amount: [
        prune.l1_unstructured(module, name, amount=amount) for module, name in params
    ],
    "layerwise_random": lambda model, params, amount: [
        prune.random_unstructured(module, name, amount=amount)
        for module, name in params
    ],
    "structured_l2_channels": lambda model, params, amount: [
        prune.ln_structured(module, name, amount=amount, n=2, dim=0)
        for module, name in params
    ],
    "semi_structured_nm_weightnorm": lambda model, params, amount: _ao_sparsify_model_with_config(
        model,
        WeightNormSparsifier(
            sparsity_level=1.0,
            sparse_block_shape=(1, 4),
            zeros_per_block=int(max(0, min(4, round(amount * 4)))),
        ),
    ),
    "wanda_unstructured": lambda model, params, amount: _wanda_sparsify(model, amount),
    "taylor_unstructured": _taylor_unstructured_no_bias,
}

keys_to_omit = []
for key in keys_to_omit:
    PRUNING_STRATEGIES.pop(key)


def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def estimate_IZY_and_IXZ(model, loader):
    return estimate_mi_zy(model, loader, DEVICE, mi_config=None), estimate_mi_zx(
        model, loader, DEVICE, cfg, mi_config=None
    )


def error_breakdown_by_true_class(model, loader, num_classes=10):
    model.eval()
    total_per_class = torch.zeros(num_classes, dtype=torch.long)
    errors_per_class = torch.zeros(num_classes, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            wrong = pred != y
            for c in range(num_classes):
                yc = y == c
                total_per_class[c] += yc.sum().item()
                errors_per_class[c] += (yc & wrong).sum().item()
    total_errors = int(errors_per_class.sum().item())
    if total_errors > 0:
        error_share = (errors_per_class.float() / total_errors).tolist()
    else:
        error_share = [0.0] * num_classes
    class_error_rate = (
        errors_per_class.float() / total_per_class.clamp_min(1).float()
    ).tolist()
    return {
        "errors_per_class": [int(v) for v in errors_per_class.tolist()],
        "total_per_class": [int(v) for v in total_per_class.tolist()],
        "total_errors": total_errors,
        "error_share": error_share,
        "class_error_rate": class_error_rate,
    }


def main():
    transform = transforms.ToTensor()
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    subset_test_set = torch.utils.data.Subset(
        test_ds, torch.randperm(len(test_ds))[:MI_SAMPLE]
    )
    test_loader_mi = DataLoader(subset_test_set, batch_size=BATCH_SIZE, shuffle=False)
    model_base = MLP().to(DEVICE)
    model_base.load_state_dict(torch.load("mlp_mnist.pth", map_location=DEVICE))
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
    os.makedirs("pruning_plots_no_last_layer", exist_ok=True)
    all_results = {}
    for strategy_name, strategy_fn in PRUNING_STRATEGIES.items():
        accs, izy_list, ixz_list = [], [], []
        err_counts_list = []
        err_shares_list = []
        totals_per_class = None
        print(f"\n=== Pruning strategy: {strategy_name} ===")
        for ratio in ratios:
            model = copy.deepcopy(model_base)
            parameters_to_prune = [
                (model.fc1, "weight"),
                (model.fc2, "weight"),
            ]
            if ratio > 0.0:
                strategy_fn(model, parameters_to_prune, ratio)
            acc = evaluate_accuracy(model, test_loader)
            breakdown = error_breakdown_by_true_class(
                model, test_loader, num_classes=10
            )
            I_ZY, I_XZ = estimate_IZY_and_IXZ(model, test_loader_mi)
            if totals_per_class is None:
                totals_per_class = breakdown["total_per_class"]
            accs.append(acc)
            izy_list.append(I_ZY)
            ixz_list.append(I_XZ)
            err_counts_list.append(breakdown["errors_per_class"])
            err_shares_list.append(breakdown["error_share"])
            shares_pct = [f"{100*s:.1f}%" for s in breakdown["error_share"]]
            shares_str = ", ".join(
                [f"class {i}: {p}" for i, p in enumerate(shares_pct)]
            )
            print(
                f"Prune ratio: {ratio:.2f} | "
                f"Test Acc: {acc*100:6.2f}% | "
                f"I(Z;Y): {I_ZY:7.3f} bits | I(X;Z): {I_XZ:7.3f} bits"
            )
            print(
                f"  Error share by true class (percent of all mistakes): {shares_str}"
            )
        all_results[strategy_name] = {
            "ratios": ratios,
            "accs": accs,
            "izy": izy_list,
            "ixz": ixz_list,
            "err_counts": err_counts_list,
            "err_shares": err_shares_list,
            "totals": totals_per_class,
        }
    plt.figure(figsize=(8, 6))
    for strategy_name, results in all_results.items():
        plt.plot(results["ratios"], results["accs"], marker="o", label=strategy_name)
    plt.xlabel("Prune Ratio")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Prune Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(
        "pruning_plots_no_last_layer", "acc_vs_ratio_all_methods.png"
    )
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")
    plt.figure(figsize=(8, 6))
    for strategy_name, results in all_results.items():
        plt.plot(results["ratios"], results["ixz"], marker="o", label=strategy_name)
    plt.xlabel("Prune Ratio")
    plt.ylabel("I(X;Z) (bits)")
    plt.title("I(X;Z) vs Prune Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(
        "pruning_plots_no_last_layer", "ixz_vs_ratio_all_methods.png"
    )
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")
    plt.figure(figsize=(8, 6))
    for strategy_name, results in all_results.items():
        plt.plot(results["ratios"], results["izy"], marker="o", label=strategy_name)
    plt.xlabel("Prune Ratio")
    plt.ylabel("I(Z;Y) (bits)")
    plt.title("I(Z;Y) vs Prune Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(
        "pruning_plots_no_last_layer", "izy_vs_ratio_all_methods.png"
    )
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")
    for strategy_name, results in all_results.items():
        counts_arr = np.array(results["err_counts"])
        totals = (
            results["totals"]
            if results.get("totals") is not None
            else [None] * counts_arr.shape[1]
        )
        plt.figure(figsize=(9, 6))
        for c in range(counts_arr.shape[1]):
            label_total = f" (N={totals[c]})" if totals[c] is not None else ""
            plt.plot(
                results["ratios"],
                counts_arr[:, c],
                marker="o",
                label=f"class {c}{label_total}",
            )
        plt.xlabel("Prune Ratio")
        plt.ylabel("# Errors (absolute)")
        plt.title(f"Errors per Class vs Prune Ratio - {strategy_name}")
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(
            "pruning_plots_no_last_layer",
            f"errors_per_class_counts_{strategy_name}.png",
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")
    for strategy_name, results in all_results.items():
        shares_arr = np.array(results["err_shares"])
        shares_arr_pct = shares_arr * 100.0
        totals = (
            results["totals"]
            if results.get("totals") is not None
            else [None] * shares_arr_pct.shape[1]
        )
        plt.figure(figsize=(9, 6))
        for c in range(shares_arr_pct.shape[1]):
            label_total = f" (N={totals[c]})" if totals[c] is not None else ""
            plt.plot(
                results["ratios"],
                shares_arr_pct[:, c],
                marker="o",
                label=f"class {c}{label_total}",
            )
        plt.xlabel("Prune Ratio")
        plt.ylabel("Error Share (%)")
        plt.title(f"Error Share per Class vs Prune Ratio - {strategy_name}")
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(
            "pruning_plots_no_last_layer",
            f"errors_per_class_percentage_{strategy_name}.png",
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")
    for strategy_name, results in all_results.items():
        ratios = results["ratios"]
        acc_pct = 100.0 * np.array(results["accs"])
        I_vals = np.array(results["izy"])
        H_Y_bits = math.log2(10)
        theo_acc = [
            fano_upper_accuracy_from_I(I_bits, K=10, H_Y_bits=H_Y_bits)
            for I_bits in I_vals
        ]
        theo_pct = 100.0 * np.array(theo_acc)
        plt.figure(figsize=(8.5, 6))
        plt.plot(ratios, acc_pct, marker="o", label="Actual Accuracy (%)")
        plt.plot(
            ratios, theo_pct, marker="s", label="Theoretical (Fano upper bound) (%)"
        )
        plt.xlabel("Prune Ratio")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Actual vs Theoretical Accuracy - {strategy_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(
            "pruning_plots_no_last_layer",
            f"accuracy_vs_theoretical_{strategy_name}.png",
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
