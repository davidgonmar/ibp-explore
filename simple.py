import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mutual_info.estimate_mi import estimate_mi_zy, estimate_mi_zx
from omegaconf import DictConfig

import matplotlib.pyplot as plt

# ====================== config ======================
BATCH_SIZE = 128
LR = 1e-3
STEPS_EPOCH = 5
MAX_EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MI parameters
MI_SAMPLE = 200
KNN_K = 5
JITTER = 1e-6  # stronger jitter to avoid zero kNN radii
LOG_BASE = 2  # bits

# Autoencoder params
BOTTLENECK_X = 4  # X code dim
BOTTLENECK_Z = 4  # Z code dim
AE_HID_X = 256
AE_HID_Z = 256
AE_EPOCHS = 10
AE_BATCH = 256
AE_LR = 1e-3


cfg = DictConfig({"dataset": {"name": "mnist", "size": 28, "num_channels": 1}})


# ====================== model ======================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 100)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x, return_hidden=False):
        x = self.flatten(x)
        h = self.act1(self.fc1(x))
        z = self.act2(self.fc2(h))
        out = self.fc3(z)
        if return_hidden:
            return out, z
        return out


# ====================== helpers ======================
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
    """model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    """
    # print("Estimating I(Z;Y) and I(X;Z)...")
    return estimate_mi_zy(model, loader, DEVICE, mi_config=None), estimate_mi_zx(
        model, loader, DEVICE, cfg, mi_config=None
    )


# ====================== training ======================
def main():
    # data
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    subset_test_set = torch.utils.data.Subset(
        test_ds, torch.randperm(len(test_ds))[:MI_SAMPLE]
    )
    test_loader_mi = DataLoader(subset_test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = MLP().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # Lists to store metrics for plotting
    epochs_list = []
    train_acc_list = []
    test_acc_list = []
    izy_list = []
    ixz_list = []

    step = 0
    epoch = 0
    while epoch < MAX_EPOCHS:
        for xb, yb in train_loader:
            model.train()
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            step += 1

            if step % STEPS_EPOCH == 0:
                epoch += 1
                # Compute both train and test accuracy
                train_acc = evaluate_accuracy(model, train_loader)
                test_acc = evaluate_accuracy(model, test_loader)
                I_ZY, I_XZ = estimate_IZY_and_IXZ(model, test_loader_mi)
                print(
                    f"Epoch {epoch:03d} | Step {step:05d} | "
                    f"Train Acc: {train_acc*100:6.2f}% | Test Acc: {test_acc*100:6.2f}% | "
                    f"I(Z;Y): {I_ZY:7.3f} bits | I(X;Z): {I_XZ:7.3f} bits"
                )
                # Store metrics
                epochs_list.append(epoch)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                izy_list.append(I_ZY)
                ixz_list.append(I_XZ)
                if epoch >= MAX_EPOCHS:
                    break

    # Plotting at the end of training
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_list, train_acc_list, label="Train Accuracy")
    plt.plot(epochs_list, test_acc_list, label="Test Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title("Train and Test Accuracy over Epochs")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs_list, izy_list, label="I(Z;Y)")
    plt.plot(epochs_list, ixz_list, label="I(X;Z)")
    plt.ylabel("Mutual Information (bits)")
    plt.xlabel("Epoch")
    plt.title("Mutual Information Estimates")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
