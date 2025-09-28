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

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MI_SAMPLE = 200

cfg = DictConfig({"dataset": {"name": "mnist", "size": 28, "num_channels": 1}})


def _ao_sparsify_model_with_config(model, sparsifier):
    sparse_config = []
    for fqn, mod in model.named_modules():
        if fqn in ("fc1", "fc2", "fc3") and isinstance(mod, torch.nn.Linear):
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
        if fqn in ("fc1", "fc2", "fc3") and isinstance(mod, torch.nn.Linear):
            sparse_config.append({"tensor_fqn": f"{fqn}.weight"})
    sparsifier = WandaSparsifier(
        sparsity_level=amount,
    )
    sparsifier.prepare(model, sparse_config)
    # run model through some data to compute importance scores
    loader = DataLoader(mnist_subset, batch_size=BATCH_SIZE, shuffle=False)
    for x, _ in loader:
        x = x.to(DEVICE)
        model(x)
    sparsifier.step()
    sparsifier.squash_mask()


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
}


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


def main():
    transform = transforms.ToTensor()
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    subset_test_set = torch.utils.data.Subset(
        test_ds, torch.randperm(len(test_ds))[:MI_SAMPLE]
    )
    test_loader_mi = DataLoader(subset_test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Create a base model and load the weights once
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

    os.makedirs("pruning_plots", exist_ok=True)

    all_results = {}
    for strategy_name, strategy_fn in PRUNING_STRATEGIES.items():
        accs = []
        izy_list = []
        ixz_list = []
        print(f"\n=== Pruning strategy: {strategy_name} ===")
        for ratio in ratios:
            # Deepcopy the base model for each pruning step
            model = copy.deepcopy(model_base)

            parameters_to_prune = [
                (model.fc1, "weight"),
                (model.fc2, "weight"),
                (model.fc3, "weight"),
            ]

            if ratio > 0.0:
                strategy_fn(model, parameters_to_prune, ratio)

            acc = evaluate_accuracy(model, test_loader)
            I_ZY, I_XZ = estimate_IZY_and_IXZ(model, test_loader_mi)

            accs.append(acc)
            izy_list.append(I_ZY)
            ixz_list.append(I_XZ)

            print(
                f"Prune ratio: {ratio:.2f} | "
                f"Test Acc: {acc*100:6.2f}% | "
                f"I(Z;Y): {I_ZY:7.3f} bits | I(X;Z): {I_XZ:7.3f} bits"
            )

        all_results[strategy_name] = {
            "ratios": ratios,
            "accs": accs,
            "izy": izy_list,
            "ixz": ixz_list,
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
    plot_path = os.path.join("pruning_plots", "acc_vs_ratio_all_methods.png")
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
    plot_path = os.path.join("pruning_plots", "ixz_vs_ratio_all_methods.png")
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
    plot_path = os.path.join("pruning_plots", "izy_vs_ratio_all_methods.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
