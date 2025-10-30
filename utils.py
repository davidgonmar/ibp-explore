import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from mutual_info.estimate_mi import estimate_mi_zy, estimate_mi_zx

BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MI_SAMPLE = 4096
cfg = DictConfig({"dataset": {"name": "mnist", "size": 28, "num_channels": 1}})


def get_test_and_mi_loaders():
    transform = transforms.ToTensor()
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    subset_test_set = torch.utils.data.Subset(
        test_ds, torch.randperm(len(test_ds))[:MI_SAMPLE]
    )
    test_loader_mi = DataLoader(subset_test_set, batch_size=BATCH_SIZE, shuffle=False)

    return test_loader, test_loader_mi


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
    return (
        estimate_mi_zy(model, loader, DEVICE, mi_config=None),
        estimate_mi_zx(model, loader, DEVICE, cfg, mi_config=None),
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


def _binary_entropy_bits(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _fano_error_upper_bound(
    I_bits: float,
    K: int = 10,
    H_Y_bits: float = None,
    tol: float = 1e-7,
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


def fano_upper_accuracy_from_I(I_bits_values, K: int = 10, H_Y_bits: float = None):
    accs = []
    for I_bits in I_bits_values:
        Pe = _fano_error_upper_bound(I_bits, K=K, H_Y_bits=H_Y_bits)
        accs.append(1.0 - Pe)
    return accs


def save_results_json(results_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results_dict, f)


def load_results_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_pruning_results(results_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # results_dict looks like:
    # {
    #   "<strategy>": {
    #        "ratios": [...],
    #        "accs": [...],
    #        "izy": [...],
    #        "ixz": [...],
    #        "err_counts": [[...], ...],
    #        "err_shares": [[...], ...],
    #        "totals": [...]
    #   },
    #   ...
    # }

    # acc vs ratio
    plt.figure(figsize=(8, 6))
    for strategy_name, res in results_dict.items():
        plt.plot(res["ratios"], res["accs"], marker="o", label=strategy_name)
    plt.xlabel("Prune Ratio")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Prune Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_vs_ratio_all_methods.png"))
    plt.close()

    # ixz vs ratio
    plt.figure(figsize=(8, 6))
    for strategy_name, res in results_dict.items():
        plt.plot(res["ratios"], res["ixz"], marker="o", label=strategy_name)
    plt.xlabel("Prune Ratio")
    plt.ylabel("I(X;Z) (bits)")
    plt.title("I(X;Z) vs Prune Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ixz_vs_ratio_all_methods.png"))
    plt.close()

    # izy vs ratio
    plt.figure(figsize=(8, 6))
    for strategy_name, res in results_dict.items():
        plt.plot(res["ratios"], res["izy"], marker="o", label=strategy_name)
    plt.xlabel("Prune Ratio")
    plt.ylabel("I(Z;Y) (bits)")
    plt.title("I(Z;Y) vs Prune Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "izy_vs_ratio_all_methods.png"))
    plt.close()

    # per-class absolute errors, per strategy
    for strategy_name, res in results_dict.items():
        counts_arr = np.array(res["err_counts"])
        totals = (
            res["totals"]
            if res.get("totals") is not None
            else [None] * counts_arr.shape[1]
        )

        plt.figure(figsize=(9, 6))
        for c in range(counts_arr.shape[1]):
            label_total = f" (N={totals[c]})" if totals[c] is not None else ""
            plt.plot(
                res["ratios"],
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
        plt.savefig(
            os.path.join(
                out_dir,
                f"errors_per_class_counts_{strategy_name}.png",
            )
        )
        plt.close()

    # per-class % share of total errors, per strategy
    for strategy_name, res in results_dict.items():
        shares_arr = np.array(res["err_shares"]) * 100.0
        totals = (
            res["totals"]
            if res.get("totals") is not None
            else [None] * shares_arr.shape[1]
        )

        plt.figure(figsize=(9, 6))
        for c in range(shares_arr.shape[1]):
            label_total = f" (N={totals[c]})" if totals[c] is not None else ""
            plt.plot(
                res["ratios"],
                shares_arr[:, c],
                marker="o",
                label=f"class {c}{label_total}",
            )

        plt.xlabel("Prune Ratio")
        plt.ylabel("Error Share (%)")
        plt.title(f"Error Share per Class vs Prune Ratio - {strategy_name}")
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                out_dir,
                f"errors_per_class_percentage_{strategy_name}.png",
            )
        )
        plt.close()

    # actual acc vs Fano upper bound, per strategy
    for strategy_name, res in results_dict.items():
        ratios = res["ratios"]
        acc_pct = 100.0 * np.array(res["accs"])
        I_vals = np.array(res["izy"])
        theo_acc = fano_upper_accuracy_from_I(
            I_vals,
            K=10,
            H_Y_bits=math.log2(10),
        )
        theo_pct = 100.0 * np.array(theo_acc)

        plt.figure(figsize=(8.5, 6))
        plt.plot(ratios, acc_pct, marker="o", label="Actual Accuracy (%)")
        plt.plot(
            ratios,
            theo_pct,
            marker="s",
            label="Theoretical (Fano upper bound) (%)",
        )
        plt.xlabel("Prune Ratio")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Actual vs Theoretical Accuracy - {strategy_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                out_dir,
                f"accuracy_vs_theoretical_{strategy_name}.png",
            )
        )
        plt.close()
