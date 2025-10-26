import warnings

warnings.filterwarnings(
    "ignore",
    message=".*torch.ao.quantization is deprecated.*",
    category=DeprecationWarning,
)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from omegaconf import DictConfig
from mutual_info.estimate_mi import estimate_mi_zy, estimate_mi_zx
from models import MLP
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
import math
from itertools import product
import random

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MI_SAMPLE = 200
MAX_HETERO_COMBOS = 10

cfg = DictConfig({"dataset": {"name": "mnist", "size": 28, "num_channels": 1}})

mnist_train = datasets.MNIST(
    "./data", train=True, download=True, transform=transforms.ToTensor()
)
mnist_subset = torch.utils.data.Subset(
    mnist_train, torch.randperm(len(mnist_train))[:MI_SAMPLE]
)

_ce_loss = torch.nn.CrossEntropyLoss()

CONFIGS = [
    # {"name": "W2_A2_sym", "w_bits": 2, "a_bits": 2, "w_scheme": "per_channel_symmetric", "a_scheme": "per_tensor_symmetric", "w_sym": True, "a_sym": True},
    {
        "name": "W3_A3_sym",
        "w_bits": 3,
        "a_bits": 3,
        "w_scheme": "per_channel_symmetric",
        "a_scheme": "per_tensor_symmetric",
        "w_sym": True,
        "a_sym": True,
    },
    {
        "name": "W4_A4_sym",
        "w_bits": 4,
        "a_bits": 4,
        "w_scheme": "per_channel_symmetric",
        "a_scheme": "per_tensor_symmetric",
        "w_sym": True,
        "a_sym": True,
    },
    {
        "name": "W5_A4_sym",
        "w_bits": 5,
        "a_bits": 4,
        "w_scheme": "per_channel_symmetric",
        "a_scheme": "per_tensor_symmetric",
        "w_sym": True,
        "a_sym": True,
    },
    {
        "name": "W6_A3_sym",
        "w_bits": 6,
        "a_bits": 3,
        "w_scheme": "per_channel_symmetric",
        "a_scheme": "per_tensor_symmetric",
        "w_sym": True,
        "a_sym": True,
    },
    {
        "name": "W7_A8_mix",
        "w_bits": 7,
        "a_bits": 8,
        "w_scheme": "per_channel_symmetric",
        "a_scheme": "per_tensor_affine",
        "w_sym": True,
        "a_sym": False,
    },
    {
        "name": "W8_A6_mix",
        "w_bits": 8,
        "a_bits": 6,
        "w_scheme": "per_channel_symmetric",
        "a_scheme": "per_tensor_affine",
        "w_sym": True,
        "a_sym": False,
    },
    {
        "name": "W4_A6_asymA",
        "w_bits": 4,
        "a_bits": 6,
        "w_scheme": "per_channel_symmetric",
        "a_scheme": "per_tensor_affine",
        "w_sym": True,
        "a_sym": False,
    },
    {
        "name": "W5_A7_asymA",
        "w_bits": 5,
        "a_bits": 7,
        "w_scheme": "per_channel_symmetric",
        "a_scheme": "per_tensor_affine",
        "w_sym": True,
        "a_sym": False,
    },
    # {"name": "W2_A8_asymA", "w_bits": 2, "a_bits": 8, "w_scheme": "per_channel_symmetric", "a_scheme": "per_tensor_affine", "w_sym": True, "a_sym": False},
    # {"name": "W8_A2_sym", "w_bits": 8, "a_bits": 2, "w_scheme": "per_channel_symmetric", "a_scheme": "per_tensor_symmetric", "w_sym": True, "a_sym": True},
    {
        "name": "W6_A5_mix",
        "w_bits": 6,
        "a_bits": 5,
        "w_scheme": "per_channel_symmetric",
        "a_scheme": "per_tensor_affine",
        "w_sym": True,
        "a_sym": False,
    },
    {
        "name": "W3_A7_mix",
        "w_bits": 3,
        "a_bits": 7,
        "w_scheme": "per_channel_symmetric",
        "a_scheme": "per_tensor_affine",
        "w_sym": True,
        "a_sym": False,
    },
    {
        "name": "W7_A3_sym",
        "w_bits": 7,
        "a_bits": 3,
        "w_scheme": "per_channel_symmetric",
        "a_scheme": "per_tensor_symmetric",
        "w_sym": True,
        "a_sym": True,
    },
    {
        "name": "W5_A5_sym",
        "w_bits": 5,
        "a_bits": 5,
        "w_scheme": "per_channel_symmetric",
        "a_scheme": "per_tensor_symmetric",
        "w_sym": True,
        "a_sym": True,
    },
]


def _make_ranges(bits, symmetric):
    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1
        dtype = torch.qint8
    else:
        qmin = 0
        qmax = (2**bits) - 1
        dtype = torch.quint8
    return qmin, qmax, dtype


def _qscheme_from_str(s):
    if s == "per_tensor_affine":
        return torch.per_tensor_affine
    if s == "per_tensor_symmetric":
        return torch.per_tensor_symmetric
    if s == "per_channel_symmetric":
        return torch.per_channel_symmetric
    return torch.per_tensor_symmetric


def _make_qconfig_from_conf(conf):
    w_qmin, w_qmax, w_dtype = _make_ranges(conf["w_bits"], conf["w_sym"])
    a_qmin, a_qmax, a_dtype = _make_ranges(conf["a_bits"], conf["a_sym"])
    a_qscheme = _qscheme_from_str(conf["a_scheme"])
    w_qscheme = _qscheme_from_str(conf["w_scheme"])
    act_fq = FakeQuantize.with_args(
        observer=MinMaxObserver,
        dtype=a_dtype,
        qscheme=a_qscheme,
        quant_min=a_qmin,
        quant_max=a_qmax,
        reduce_range=False,
    )
    if w_qscheme == torch.per_channel_symmetric:
        w_fq = FakeQuantize.with_args(
            observer=PerChannelMinMaxObserver,
            dtype=w_dtype,
            qscheme=w_qscheme,
            ch_axis=0,
            quant_min=w_qmin,
            quant_max=w_qmax,
            reduce_range=False,
        )
    else:
        w_fq = FakeQuantize.with_args(
            observer=MinMaxObserver,
            dtype=w_dtype,
            qscheme=w_qscheme,
            quant_min=w_qmin,
            quant_max=w_qmax,
            reduce_range=False,
        )
    return torch.ao.quantization.QConfig(activation=act_fq, weight=w_fq)


def _apply_layerwise_fakequant(model, conf_tuple):
    m = copy.deepcopy(model).cpu()
    m.train()
    m.qconfig = None
    m.fc1.qconfig = _make_qconfig_from_conf(conf_tuple[0])
    m.fc2.qconfig = _make_qconfig_from_conf(conf_tuple[1])
    m.fc3.qconfig = _make_qconfig_from_conf(conf_tuple[2])
    torch.ao.quantization.prepare_qat(m, inplace=True)
    calib_loader = DataLoader(mnist_subset, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for x, _ in calib_loader:
            m(x)
    m.eval()
    return m


def _model_device_and_dtype(model):
    p = next(model.parameters(), None)
    if p is None:
        return torch.device("cpu"), torch.float32
    return p.device, p.dtype


def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            mdev, mdtype = _model_device_and_dtype(model)
            x = x.to(device=mdev)
            y = y.to(mdev)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def estimate_IZY_and_IXZ(model, loader):
    mdev, _ = _model_device_and_dtype(model)
    return estimate_mi_zy(model, loader, mdev, mi_config=None), estimate_mi_zx(
        model, loader, mdev, cfg, mi_config=None
    )


def error_breakdown_by_true_class(model, loader, num_classes=10):
    model.eval()
    total_per_class = torch.zeros(num_classes, dtype=torch.long)
    errors_per_class = torch.zeros(num_classes, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            mdev, _ = _model_device_and_dtype(model)
            x = x.to(device=mdev)
            y = y.to(mdev)
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
        if g(mid) > 0.0:
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

    os.makedirs("quantization_plots", exist_ok=True)

    accs, izy_list, ixz_list = [], [], []
    err_counts_list = []
    err_shares_list = []
    totals_per_class = None
    labels = []

    conf_iter = product(CONFIGS, repeat=3)

    if MAX_HETERO_COMBOS is not None:
        # ramdom selection
        conf_iter = random.sample(list(conf_iter), MAX_HETERO_COMBOS)

    print("\n=== Quantization strategy: fakequant_layerwise_heterogeneous ===")
    for conf_tuple in conf_iter:
        model = copy.deepcopy(model_base)
        model = _apply_layerwise_fakequant(model, conf_tuple)
        label = (
            f"{conf_tuple[0]['name']}|{conf_tuple[1]['name']}|{conf_tuple[2]['name']}"
        )

        acc = evaluate_accuracy(model, test_loader)
        breakdown = error_breakdown_by_true_class(model, test_loader, num_classes=10)
        I_ZY, I_XZ = estimate_IZY_and_IXZ(model, test_loader_mi)

        if totals_per_class is None:
            totals_per_class = breakdown["total_per_class"]

        accs.append(acc)
        izy_list.append(I_ZY)
        ixz_list.append(I_XZ)
        err_counts_list.append(breakdown["errors_per_class"])
        err_shares_list.append(breakdown["error_share"])
        labels.append(label)

        shares_pct = [f"{100*s:.1f}%" for s in breakdown["error_share"]]
        shares_str = ", ".join([f"class {i}: {p}" for i, p in enumerate(shares_pct)])
        print(
            f"Config: {label} | Test Acc: {acc*100:6.2f}% | I(Z;Y): {I_ZY:7.3f} bits | I(X;Z): {I_XZ:7.3f} bits"
        )
        print(f"  Error share by true class (percent of all mistakes): {shares_str}")

    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    plt.plot(x, accs, marker="o", label="fakequant_layerwise_heterogeneous")
    plt.xlabel("Quantization Config (fc1|fc2|fc3)")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Heterogeneous Layerwise Quantization Config")
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join("quantization_plots", "acc_vs_configs_hetero.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

    plt.figure(figsize=(12, 6))
    plt.plot(x, ixz_list, marker="o", label="fakequant_layerwise_heterogeneous")
    plt.xlabel("Quantization Config (fc1|fc2|fc3)")
    plt.ylabel("I(X;Z) (bits)")
    plt.title("I(X;Z) vs Heterogeneous Layerwise Quantization Config")
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join("quantization_plots", "ixz_vs_configs_hetero.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

    plt.figure(figsize=(12, 6))
    plt.plot(x, izy_list, marker="o", label="fakequant_layerwise_heterogeneous")
    plt.xlabel("Quantization Config (fc1|fc2|fc3)")
    plt.ylabel("I(Z;Y) (bits)")
    plt.title("I(Z;Y) vs Heterogeneous Layerwise Quantization Config")
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join("quantization_plots", "izy_vs_configs_hetero.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

    counts_arr = np.array(err_counts_list)
    totals = (
        totals_per_class
        if totals_per_class is not None
        else [None] * counts_arr.shape[1]
    )
    plt.figure(figsize=(12, 6))
    for c in range(counts_arr.shape[1]):
        label_total = f" (N={totals[c]})" if totals[c] is not None else ""
        plt.plot(x, counts_arr[:, c], marker="o", label=f"class {c}{label_total}")
    plt.xlabel("Quantization Config (fc1|fc2|fc3)")
    plt.ylabel("# Errors (absolute)")
    plt.title("Errors per Class vs Heterogeneous Layerwise Quantization Config")
    plt.xticks(x, labels, rotation=90)
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join("quantization_plots", "errors_per_class_counts_hetero.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

    shares_arr = np.array(err_shares_list)
    shares_arr_pct = shares_arr * 100.0
    totals = (
        totals_per_class
        if totals_per_class is not None
        else [None] * shares_arr_pct.shape[1]
    )
    plt.figure(figsize=(12, 6))
    for c in range(shares_arr_pct.shape[1]):
        label_total = f" (N={totals[c]})" if totals[c] is not None else ""
        plt.plot(x, shares_arr_pct[:, c], marker="o", label=f"class {c}{label_total}")
    plt.xlabel("Quantization Config (fc1|fc2|fc3)")
    plt.ylabel("Error Share (%)")
    plt.title("Error Share per Class vs Heterogeneous Layerwise Quantization Config")
    plt.xticks(x, labels, rotation=90)
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(
        "quantization_plots", "errors_per_class_percentage_hetero.png"
    )
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

    acc_pct = 100.0 * np.array(accs)
    I_vals = np.array(izy_list)
    H_Y_bits = math.log2(10)
    theo_acc = [
        fano_upper_accuracy_from_I(I_bits, K=10, H_Y_bits=H_Y_bits) for I_bits in I_vals
    ]
    theo_pct = 100.0 * np.array(theo_acc)

    plt.figure(figsize=(12, 6))
    plt.plot(x, acc_pct, marker="o", label="Actual Accuracy (%)")
    plt.plot(x, theo_pct, marker="s", label="Theoretical (Fano upper bound) (%)")
    plt.xlabel("Quantization Config (fc1|fc2|fc3)")
    plt.ylabel("Accuracy (%)")
    plt.title("Actual vs Theoretical Accuracy (Fano) — Heterogeneous Layerwise")
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join("quantization_plots", "accuracy_vs_theoretical_hetero.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
