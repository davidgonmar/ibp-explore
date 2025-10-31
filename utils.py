import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import json
import os
from omegaconf import DictConfig
from mutual_info.estimate_mi import estimate_mi_zy, estimate_mi_zx


BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MI_SAMPLE = 4096


def get_test_and_mi_loaders(dsname="mnist"):
    dsname = dsname.lower()
    if dsname == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        test_ds = datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=transform,
        )
    elif dsname == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_ds = datasets.CIFAR10(
            "./data",
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset name: {dsname}")

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    subset_test_set = torch.utils.data.Subset(
        test_ds,
        torch.randperm(len(test_ds))[:MI_SAMPLE],
    )
    test_loader_mi = DataLoader(
        subset_test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
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


def estimate_IZY_and_IXZ(model, dsname, loader):
    if dsname.lower() == "mnist":
        cfg = DictConfig({"dataset": {"name": "mnist", "size": 28, "num_channels": 1}})
    elif dsname.lower() == "cifar10":
        cfg = DictConfig(
            {"dataset": {"name": "cifar10", "size": 32, "num_channels": 3}}
        )
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


def fano_upper_accuracy_from_I(
    I_bits_values,
    K: int = 10,
    H_Y_bits: float = None,
):
    accs = []
    for I_bits in I_bits_values:
        Pe = _fano_error_upper_bound(
            I_bits,
            K=K,
            H_Y_bits=H_Y_bits,
        )
        accs.append(1.0 - Pe)
    return accs


def save_results_json(results_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results_dict, f)


def load_results_json(path):
    with open(path, "r") as f:
        return json.load(f)


import torch


def evaluate_with_classifier(
    model,
    loader,
    classifiers="linear",
    num_classes=10,
):
    def build_classifier(spec, in_dim, num_classes):
        if isinstance(spec, str):
            s = spec.lower()
            if s == "linear":
                clf = torch.nn.Linear(in_dim, num_classes).to(DEVICE)
            elif s == "mlp":
                hidden_dim = 128
                clf = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, num_classes),
                ).to(DEVICE)
            else:
                raise ValueError(
                    f"Unknown classifier spec '{spec}'. Use 'linear', 'mlp', or pass an nn.Module."
                )
            name = s
        elif isinstance(spec, torch.nn.Module):
            clf = spec.to(DEVICE)
            name = clf.__class__.__name__.lower()
        else:
            raise TypeError("classifier must be 'linear', 'mlp', or a torch.nn.Module.")
        return name, clf

    def compute_accuracy(logits, Y):
        preds = logits.argmax(dim=1)
        correct = (preds == Y).sum().item()
        total = Y.size(0)
        return correct / total if total > 0 else 0.0

    model.eval()
    feats_list = []
    ys_list = []
    decoder_logits_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            out, z = model(x, return_hidden=True)
            feats_list.append(z)
            ys_list.append(y)
            decoder_logits_list.append(out)

    X = torch.cat(feats_list, dim=0)
    Y = torch.cat(ys_list, dim=0)
    decoder_logits = torch.cat(decoder_logits_list, dim=0)

    results = {}
    results["original"] = {"accuracy": compute_accuracy(decoder_logits, Y)}

    if not isinstance(classifiers, (list, tuple)):
        classifiers = [classifiers]

    for spec in classifiers:
        name, clf = build_classifier(spec, X.size(1), num_classes)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
        loss_fn = torch.nn.CrossEntropyLoss()
        clf.train()
        for _ in range(5):
            opt.zero_grad()
            logits = clf(X)
            loss = loss_fn(logits, Y)
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            logits = clf(X)
        results[name] = {"accuracy": compute_accuracy(logits, Y)}

    return results
