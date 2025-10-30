import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import prune
from torch.ao.pruning import WeightNormSparsifier
from torchao.sparsity import WandaSparsifier
import copy
from sparse.taylor_prune import prune_taylor_unstructured
from models import MLP
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
        if fqn in ("fc1", "fc2", "fc3") and isinstance(mod, torch.nn.Linear):
            sparse_config.append({"tensor_fqn": f"{fqn}.weight"})
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()
    sparsifier.squash_mask()


def _wanda_sparsify_model_with_config(model, amount, calib_subset):
    sparsifier = WandaSparsifier(sparsity_level=amount)
    sparse_config = []
    for fqn, mod in model.named_modules():
        if fqn in ("fc1", "fc2", "fc3") and isinstance(mod, torch.nn.Linear):
            sparse_config.append({"tensor_fqn": f"{fqn}.weight"})
    sparsifier.prepare(model, sparse_config)

    calib_loader = DataLoader(calib_subset, batch_size=BATCH_SIZE, shuffle=False)
    for x, _ in calib_loader:
        x = x.to(DEVICE)
        model(x)
    sparsifier.step()
    sparsifier.squash_mask()


def _taylor_unstructured(model, amount, calib_subset):
    ce_loss = torch.nn.CrossEntropyLoss()
    calib_loader = DataLoader(calib_subset, batch_size=BATCH_SIZE, shuffle=False)
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


def fit_probe(model, loader):
    feats_list = []
    ys_list = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            _, z = model(x, return_hidden=True)
            feats_list.append(z)
            ys_list.append(y)
    X = torch.cat(feats_list, dim=0)
    Y = torch.cat(ys_list, dim=0)

    clf = torch.nn.Linear(X.size(1), 10).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    clf.train()
    for _ in range(5):
        opt.zero_grad()
        logits = clf(X)
        loss = loss_fn(logits, Y)
        loss.backward()
        opt.step()

    return clf


def evaluate_with_probe(model, loader, probe_clf):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            _, z = model(x, return_hidden=True)
            pred = probe_clf(z).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def error_breakdown_with_probe(model, loader, probe_clf, num_classes=10):
    model.eval()
    total_per_class = torch.zeros(num_classes, dtype=torch.long)
    errors_per_class = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            _, z = model(x, return_hidden=True)
            pred = probe_clf(z).argmax(dim=1)
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

    return {
        "errors_per_class": [int(v) for v in errors_per_class.tolist()],
        "total_per_class": [int(v) for v in total_per_class.tolist()],
        "total_errors": total_errors,
        "error_share": error_share,
    }


PRUNING_STRATEGIES = {
    "global_l1": lambda model, params, amount: prune.global_unstructured(
        params, pruning_method=prune.L1Unstructured, amount=amount
    ),
    "layerwise_l1": lambda model, params, amount: [
        prune.l1_unstructured(m, name, amount=amount) for (m, name) in params
    ],
    "semi_structured_nm_weightnorm": lambda model, params, amount: _ao_sparsify_model_with_config(
        model,
        WeightNormSparsifier(
            sparsity_level=1.0,
            sparse_block_shape=(1, 4),
            zeros_per_block=int(max(0, min(4, round(amount * 4)))),
        ),
    ),
}


def main():
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    test_loader, test_loader_mi = get_test_and_mi_loaders()

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

    all_results = {}

    for strategy_name, strategy_fn in PRUNING_STRATEGIES.items():
        accs = []
        izy_list = []
        ixz_list = []
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
                strategy_fn(model, parameters_to_prune, ratio)

            probe_clf = fit_probe(model, train_loader)

            acc = evaluate_with_probe(model, test_loader, probe_clf)
            breakdown = error_breakdown_with_probe(
                model, test_loader, probe_clf, num_classes=10
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
                f"Probe Acc: {acc*100:6.2f}% | "
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
            "err_counts": err_counts_list,
            "err_shares": err_shares_list,
            "totals": totals_per_class,
        }

    save_results_json(all_results, "results/pruning_with_probe.json")


if __name__ == "__main__":
    main()
