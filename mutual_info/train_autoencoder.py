import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig

from .autoencoder import (
    Autoencoder,
    ConvDecoder,
    ConvEncoder,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class MIEstimationCompressionConfig:
    latent_dim: int = 6
    num_epochs: int = 200  # Number of epochs for training the autoencoder.
    batch_size: int = 256  # Batch size for training the autoencoder.
    optimizer: str = "adam"  # Following the original code.
    learning_rate: float = 1e-3
    loss_fn: str = "l1"


def train_autoencoder(
    cfg: DictConfig,
    mi_config: MIEstimationCompressionConfig,
    autoencoder: Autoencoder,
    device: torch.device,
    num_epochs_override: int = None,
) -> None:
    if cfg.dataset.name == "mnist":
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.MNIST(
            root="./data",
            train=True,
            transform=transform,
            download=True,
        )

        test_dataset = datasets.MNIST(
            root="./data",
            train=False,
            transform=transform,
            download=True,
        )
    elif cfg.dataset.name == "cifar10":
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            transform=transform,
            download=True,
        )

        test_dataset = datasets.CIFAR10(
            root="./data",
            train=False,
            transform=transform,
            download=True,
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=mi_config.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=mi_config.batch_size,
        shuffle=False,
    )

    if mi_config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            autoencoder.parameters(),
            lr=mi_config.learning_rate,
        )
    else:
        msg = f"Optimizer {mi_config.optimizer} is not supported."
        raise NotImplementedError(msg)

    if mi_config.loss_fn == "l1":
        loss_fn = torch.nn.L1Loss()
    elif mi_config.loss_fn == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        msg = f"Loss function {mi_config.loss_fn} is not supported."
        raise NotImplementedError(msg)

    num_epochs = (
        num_epochs_override if num_epochs_override is not None else mi_config.num_epochs
    )

    for epoch in range(num_epochs):
        total_loss = 0.0
        autoencoder.train()
        for data, target in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = autoencoder(data)
            loss = loss_fn(pred, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(
            "Epoch [%d/%d], Loss: %.4f",
            epoch + 1,
            num_epochs,
            total_loss / len(train_loader),
        )

        autoencoder.eval()
        with torch.no_grad():
            total_test_loss = 0.0
            for data, target in test_loader:
                data = data.to(device)
                pred = autoencoder(data)
                loss = loss_fn(pred, data)
                total_test_loss += loss.item()
            logger.info(
                "Test Loss: %.4f",
                total_test_loss / len(test_loader),
            )


def prepare_input_autoencoder(
    cfg: DictConfig,
    mi_config: MIEstimationCompressionConfig,
    device: torch.device,
    finetune: bool = True,
) -> Autoencoder:
    input_ae_path = (
        Path(".")
        / "saved_models"
        / "autoencoder"
        / f"{cfg.dataset.name}_{mi_config.num_epochs}_{mi_config.latent_dim}.pth"
    )

    input_ae = Autoencoder(
        encoder=ConvEncoder(
            input_size=cfg.dataset.size,
            input_channels=cfg.dataset.num_channels,
            latent_dim=mi_config.latent_dim,
        ),
        decoder=ConvDecoder(
            input_size=cfg.dataset.size,
            input_channels=cfg.dataset.num_channels,
            latent_dim=mi_config.latent_dim,
        ),
    )
    input_ae.to(device)

    if input_ae_path.exists():
        logger.info("Loading pre-trained autoencoder for X...")
        input_ae.load_state_dict(torch.load(input_ae_path))
        if finetune:
            logger.info("Finetuning loaded autoencoder for a few steps...")
            train_autoencoder(cfg, mi_config, input_ae, device, num_epochs_override=2)
    else:
        logger.info("Training autoencoder for X...")
        train_autoencoder(cfg, mi_config, input_ae, device)
        input_ae_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(input_ae.state_dict(), input_ae_path)

    return input_ae
