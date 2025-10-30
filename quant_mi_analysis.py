# remove warnings
import warnings

warnings.filterwarnings("ignore")

import copy
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
)

from models import MLP
from utils import evaluate_with_classifier
from utils import (
    DEVICE,
    BATCH_SIZE,
    get_test_and_mi_loaders,
    estimate_IZY_and_IXZ,
    save_results_json,
)


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

    return torch.ao.quantization.QConfig(
        activation=act_fq,
        weight=w_fq,
    )


def apply_layerwise_fakequant(model, conf_tuple, calib_loader):
    m = copy.deepcopy(model).cpu()
    m.train()
    m.qconfig = None
    m.fc1.qconfig = _make_qconfig_from_conf(conf_tuple[0])
    m.fc2.qconfig = _make_qconfig_from_conf(conf_tuple[1])
    m.fc3.qconfig = _make_qconfig_from_conf(conf_tuple[2])

    torch.ao.quantization.prepare_qat(m, inplace=True)

    with torch.no_grad():
        for x, _ in calib_loader:
            x = x.to("cpu")
            m(x)

    m.eval()
    return m.to(DEVICE)


BASE_LAYER_SET = [
    {
        "w_bits": 3,
        "w_sym": True,
        "w_scheme": "per_channel_symmetric",
        "a_bits": 3,
        "a_sym": True,
        "a_scheme": "per_tensor_symmetric",
    },
    {
        "w_bits": 4,
        "w_sym": True,
        "w_scheme": "per_channel_symmetric",
        "a_bits": 4,
        "a_sym": False,
        "a_scheme": "per_tensor_affine",
    },
    {
        "w_bits": 8,
        "w_sym": True,
        "w_scheme": "per_channel_symmetric",
        "a_bits": 8,
        "a_sym": False,
        "a_scheme": "per_tensor_affine",
    },
]


def build_configs():
    configs = []
    for c1 in BASE_LAYER_SET:
        for c2 in BASE_LAYER_SET:
            for c3 in BASE_LAYER_SET:
                configs.append((c1, c2, c3))
    return configs


def _baseline_size_bits(model):
    total_params = 0
    for m in (model.fc1, model.fc2, model.fc3):
        total_params += m.weight.numel() + m.bias.numel()
    return total_params * 32


def _compression_ratio_from_conf(conf_tuple, base_model, baseline_bits):
    layers = [base_model.fc1, base_model.fc2, base_model.fc3]
    total_bits = 0
    for layer, c in zip(layers, conf_tuple):
        total_bits += layer.weight.numel() * c["w_bits"]
        total_bits += layer.bias.numel() * 32
    return total_bits / baseline_bits


def main():
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform,
    )
    calib_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader, test_loader_mi = get_test_and_mi_loaders()

    base_model = MLP().to(DEVICE)
    base_model.load_state_dict(torch.load("mlp_mnist.pth", map_location=DEVICE))

    baseline_bits = _baseline_size_bits(base_model)

    CONFIGS = build_configs()
    N_CONFIGS = 5
    CONFIGS = random.sample(CONFIGS, N_CONFIGS)

    labels = []
    accs = []
    izy_list = []
    ixz_list = []
    compr_list = []
    err_counts_list = []
    err_shares_list = []
    totals_per_class = None

    for conf_tuple in CONFIGS:
        label_parts = []
        for c in conf_tuple:
            lb = (
                f"W{c['w_bits']}{'_sym' if c['w_sym'] else ''}"
                f"-A{c['a_bits']}{'_sym' if c['a_sym'] else ''}"
            )
            label_parts.append(lb)
        label = "|".join(label_parts)
        labels.append(label)

        q_model = apply_layerwise_fakequant(
            base_model,
            conf_tuple,
            calib_loader,
        )

        eval_info, clf = evaluate_with_classifier(
            q_model,
            test_loader,
            classifier="linear",
            num_classes=10,
        )
        acc = eval_info["accuracy"]

        breakdown = {
            "errors_per_class": eval_info["errors_per_class"],
            "total_per_class": eval_info["total_per_class"],
            "total_errors": eval_info["total_errors"],
            "error_share": eval_info["error_share"],
        }

        I_ZY, I_XZ = estimate_IZY_and_IXZ(q_model, test_loader_mi)

        if totals_per_class is None:
            totals_per_class = breakdown["total_per_class"]

        accs.append(acc)
        izy_list.append(I_ZY)
        ixz_list.append(I_XZ)
        compr_val = _compression_ratio_from_conf(
            conf_tuple,
            base_model,
            baseline_bits,
        )
        compr_list.append(compr_val)
        err_counts_list.append(breakdown["errors_per_class"])
        err_shares_list.append(breakdown["error_share"])

        shares_pct = [f"{100 * s:.1f}%" for s in breakdown["error_share"]]
        shares_str = ", ".join([f"class {i}: {p}" for i, p in enumerate(shares_pct)])
        print(
            f"{label} | "
            f"Probe Acc: {acc * 100:6.2f}% | "
            f"I(Z;Y): {I_ZY:7.3f} bits | I(X;Z): {I_XZ:7.3f} bits"
        )
        print(" Error share by true class (percent of all mistakes): " f"{shares_str}")

    out = {
        "labels": labels,
        "accs": accs,
        "izy": izy_list,
        "ixz": ixz_list,
        "compr": compr_list,
        "err_counts": err_counts_list,
        "err_shares": err_shares_list,
        "totals": totals_per_class,
    }
    save_results_json(out, "results/quant_analysis.json")


if __name__ == "__main__":
    main()
