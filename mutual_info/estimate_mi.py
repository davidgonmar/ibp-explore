import torch
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from torch import nn
import math
import logging
from pathlib import Path

from .kl_estimator import MIEstimator
from .train_autoencoder import (
    MIEstimationCompressionConfig,
    prepare_input_autoencoder,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        hidden_dim = max(in_dim // 2, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureDecoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        hidden_dim = max(in_dim // 2, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FeatureAutoencoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = FeatureEncoder(in_dim=in_dim, latent_dim=latent_dim)
        self.decoder = FeatureDecoder(in_dim=in_dim, latent_dim=latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec


def prepare_feature_autoencoder(
    features: torch.Tensor,
    mi_config: MIEstimationCompressionConfig,
    device: torch.device,
    finetune: bool = True,
) -> FeatureAutoencoder:
    model_path = (
        Path(".")
        / "saved_models"
        / "feature_autoencoder"
        / f"{features.size(1)}_{mi_config.num_epochs}_{mi_config.latent_dim}.pth"
    )

    ae = FeatureAutoencoder(
        in_dim=features.size(1),
        latent_dim=mi_config.latent_dim,
    ).to(device)

    if model_path.exists():
        logger.info("Loading pre-trained autoencoder for Z...")
        ae.load_state_dict(torch.load(model_path))
        if finetune:
            logger.info("Finetuning loaded autoencoder for a few steps...")
            train_feature_autoencoder(
                features, mi_config, ae, device, num_epochs_override=2
            )
    else:
        logger.info("Training autoencoder for Z...")
        train_feature_autoencoder(features, mi_config, ae, device)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ae.state_dict(), model_path)

    return ae


class FeatureAutoencoderConfig:
    latent_dim: int = 4
    num_epochs: int = 30  # Number of epochs for training the autoencoder.
    batch_size: int = 256  # Batch size for training the autoencoder.
    optimizer: str = "adam"  # Following the original code.
    learning_rate: float = 1e-3
    loss_fn: str = "l1"


def train_feature_autoencoder(
    features: torch.Tensor,
    mi_config: FeatureAutoencoderConfig,
    autoencoder: FeatureAutoencoder,
    device: torch.device,
    num_epochs_override: int = None,
) -> None:
    dataset = torch.utils.data.TensorDataset(features.detach().clone())
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=mi_config.batch_size,
        shuffle=True,
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
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            pred = autoencoder(batch_x)
            loss = loss_fn(pred, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(
            "Feature AE Epoch [%d/%d], Loss: %.4f",
            epoch + 1,
            num_epochs,
            total_loss / len(loader),
        )


def normalize_data(x: torch.Tensor) -> torch.Tensor:
    """Normalize data to have zero mean and unit variance.

    Args:
        x (torch.Tensor): Input data of shape (n_samples, n_features).

    """
    x = x - x.mean(dim=0, keepdim=True)
    cov_matrix = torch.cov(x.T)
    lower_matrix = torch.linalg.cholesky(
        cov_matrix + 1e-6 * torch.eye(cov_matrix.size(0)).to(cov_matrix.device)
    )
    # Solve lower_matrix @ z = x.T for z
    return torch.linalg.solve_triangular(lower_matrix, x.T, upper=False).T


def estimate_mi_zx(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: DictConfig,
    mi_configx: MIEstimationCompressionConfig = None,
    mi_configz: FeatureAutoencoderConfig = None,
    use_autoencoder: bool = True,
) -> float:
    # return 0.0
    if mi_configz is None:
        mi_configz = FeatureAutoencoderConfig()
    if mi_configx is None:
        mi_configx = MIEstimationCompressionConfig()
    model.eval()

    input_ae = prepare_input_autoencoder(
        cfg=cfg, mi_config=mi_configx, device=device, finetune=True
    )
    input_ae.eval()
    x_compressed = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)  # noqa: PLW2901
            x_compressed.append(input_ae.encode(data))
    x_compressed = torch.cat(x_compressed, dim=0)
    x_compressed = normalize_data(x_compressed)

    features = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)  # noqa: PLW2901
            out, repres = model(data, return_hidden=True)
            features.append(repres)
    features = torch.cat(features, dim=0)

    if use_autoencoder:
        feature_ae = prepare_feature_autoencoder(
            features=features,
            mi_config=mi_configz,
            device=device,
            finetune=True,
        )
        feature_ae.eval()
        with torch.no_grad():
            z_codes = feature_ae.encode(features.to(device))
        z_compressed = z_codes.cpu()
    else:
        z_compressed = PCA(n_components=mi_configz.latent_dim).fit_transform(
            features.cpu().numpy(),
        )
        z_compressed = torch.from_numpy(z_compressed)

    z_compressed = normalize_data(z_compressed)

    mi_zx_estimator = MIEstimator(
        entropy_estimator_params={
            "method": "KL",
            "functional_params": {"k_neighbors": 5},
        },
    )
    return mi_zx_estimator.fit_estimate(z_compressed, x_compressed) * math.log2(
        math.e,
    )  # nats to bits


def estimate_mi_zy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mi_config: MIEstimationCompressionConfig = None,
    use_autoencoder: bool = True,
) -> float:
    if mi_config is None:
        mi_config = MIEstimationCompressionConfig()
    model.eval()

    targets = []
    for _, target in loader:
        targets.extend(target.numpy().tolist())
    targets = torch.tensor(targets).detach().clone()
    features = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)  # noqa: PLW2901
            out, repres = model(data, return_hidden=True)
            features.append(repres)
    features = torch.cat(features, dim=0)

    if use_autoencoder:
        feature_ae = prepare_feature_autoencoder(
            features=features,
            mi_config=mi_config,
            device=device,
            finetune=True,
        )
        feature_ae.eval()
        with torch.no_grad():
            z_codes = feature_ae.encode(features.to(device))
        z_compressed = z_codes.cpu()
    else:
        z_compressed = PCA(n_components=mi_config.latent_dim).fit_transform(
            features.cpu().numpy(),
        )
        z_compressed = torch.from_numpy(z_compressed)

    z_compressed = normalize_data(z_compressed)

    mi_zy_estimator = MIEstimator(
        y_is_discrete=True,
        entropy_estimator_params={
            "method": "KL",
            "functional_params": {"k_neighbors": 5},
        },
    )
    return mi_zy_estimator.fit_estimate(z_compressed, targets) * math.log2(
        math.e
    )  # nats to bits
