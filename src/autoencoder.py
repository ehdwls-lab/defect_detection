from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvAutoencoder(nn.Module):
    """
    RGB Patch 복원을 위한 Convolutional Autoencoder.

    입력 예:
        [batch_size, 3, 64, 64]

    출력 예:
        [batch_size, 3, 64, 64]
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 32,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels는 양수여야 합니다.")

        if latent_channels <= 0:
            raise ValueError("latent_channels는 양수여야 합니다.")

        # 입력 Patch를 작은 특징 Map으로 압축
        self.encoder = nn.Sequential(
            # 64×64 → 32×32
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # 32×32 → 16×16
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # 16×16 → 8×8
            nn.Conv2d(
                in_channels=64,
                out_channels=latent_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # 압축된 특징 Map을 원래 Patch 크기로 복원
        self.decoder = nn.Sequential(
            # 8×8 → 16×16
            nn.ConvTranspose2d(
                in_channels=latent_channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # 16×16 → 32×32
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # 32×32 → 64×64
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            # 입력 Tensor가 0~1 범위이므로 출력도 0~1로 제한
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)

    def forward(self, x: Tensor) -> Tensor:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed


if __name__ == "__main__":
    # 모델의 입출력 크기를 확인하기 위한 간단한 테스트
    model = ConvAutoencoder(
        in_channels=3,
        latent_channels=32,
    )

    sample = torch.randn(4, 3, 64, 64)
    output = model(sample)

    print("입력 크기:", sample.shape)
    print("출력 크기:", output.shape)