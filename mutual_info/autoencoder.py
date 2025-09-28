# The implementation of the autoencoder model for MI estimation.
# The following code is based on https://arxiv.org/abs/2305.08013

import torch
from torch import Tensor, nn


def conv_out(h_in: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    """Calculate the output size of a convolutional layer."""
    return (h_in + 2 * padding - kernel_size) // stride + 1


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 28,
        input_channels: int = 1,
        latent_dim: int = 4,
        dropout_rate: float = 0.1,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError
        self.dropout = nn.Dropout(p=dropout_rate)

        num_channel_scale = 1 if input_channels == 1 else 2

        self.conv1 = nn.Conv2d(
            input_channels,
            8 * num_channel_scale,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            8 * num_channel_scale,
            16 * num_channel_scale,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            16 * num_channel_scale,
            32 * num_channel_scale,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        h1 = conv_out(input_size, kernel_size=3, stride=2, padding=1)
        h2 = conv_out(h1, kernel_size=3, stride=2, padding=1)
        h3 = conv_out(h2, kernel_size=3, stride=2, padding=1)
        flat_dim = 32 * num_channel_scale * h3 * h3

        self.linear1 = nn.Linear(flat_dim, 128)
        self.linear2 = nn.Linear(128, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = self.activation(self.conv1(x))
        x = self.dropout(x)
        x = self.activation(self.conv2(x))
        x = self.dropout(x)
        x = self.activation(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.linear1(x))
        return self.linear2(x)


class ConvDecoder(nn.Module):
    def __init__(
        self,
        input_size: int = 28,
        input_channels: int = 1,
        latent_dim: int = 4,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError
        num_channel_scale = 1 if input_channels == 1 else 2

        h0 = input_size
        h1 = conv_out(h0, kernel_size=3, stride=2, padding=1)
        h2 = conv_out(h1, kernel_size=3, stride=2, padding=1)
        h3 = conv_out(h2, kernel_size=3, stride=2, padding=1)
        flat_dim = 32 * num_channel_scale * h3 * h3

        # Determine output_padding for each ConvTranspose2d layer
        # stride = 2, kernel_size = 3, padding = 1
        # h_i = 2 * h_{i+1} - 1 + output_padding
        op1 = h2 - 2 * h3 + 1
        op2 = h1 - 2 * h2 + 1
        op3 = h0 - 2 * h1 + 1

        self.num_channel_scale = num_channel_scale
        self.input_size = input_size

        self.linear1 = nn.Linear(latent_dim, 128)
        self.linear2 = nn.Linear(128, flat_dim)

        self.deconv1 = nn.ConvTranspose2d(
            32 * num_channel_scale,
            16 * num_channel_scale,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=op1,
        )
        self.deconv2 = nn.ConvTranspose2d(
            16 * num_channel_scale,
            8 * num_channel_scale,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=op2,
        )
        self.deconv3 = nn.ConvTranspose2d(
            8 * num_channel_scale,
            input_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=op3,
        )

        self.num_channel_reshape = 32 * num_channel_scale
        self.size_reshape = h3

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = torch.reshape(
            x,
            (
                -1,
                self.num_channel_reshape,
                self.size_reshape,
                self.size_reshape,
            ),
        )
        x = self.activation(self.deconv1(x))
        x = self.activation(self.deconv2(x))
        return self.deconv3(x)


class Autoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        sigma: float = 0.1,
    ) -> None:
        super().__init__()
        # Encoder and decoder.
        self.encoder = encoder
        self.decoder = decoder

        # Noise.
        self.agn = AdditiveGaussianNoise(sigma=sigma, enabled_on_inference=False)

    def forward(self, x: Tensor) -> Tensor:
        latent = self.encoder(x)
        latent = self.agn(latent)
        return self.decoder(latent)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)


class AdditiveGaussianNoise(nn.Module):
    """Additive Gaussian noise layer."""

    def __init__(
        self,
        sigma: float = 0.1,
        *,
        relative_scale: bool = True,
        enabled_on_inference: bool = False,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.relative_scale = relative_scale
        self.enabled_on_inference = enabled_on_inference

    def forward(self, x: torch.tensor) -> torch.tensor:
        if (self.training or self.enabled_on_inference) and self.sigma > 0:
            scale = self.sigma * x if self.relative_scale else self.sigma
            noise = torch.randn_like(x) * scale
            return x + noise
        return x


if __name__ == "__main__":
    # Example usage
    encoder = ConvEncoder(input_size=28, input_channels=1, latent_dim=4)
    decoder = ConvDecoder(input_size=28, input_channels=1, latent_dim=4)
    autoencoder = Autoencoder(encoder, decoder)

    # Test the autoencoder with a random tensor
    test_input = torch.randn(1, 1, 28, 28)
    output = autoencoder(test_input)
    print(test_input.shape, output.shape)  # noqa: T201
