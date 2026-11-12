from __future__ import annotations

import torch
from torch import Tensor, nn

from tomato_recon.data.schemas import FeatureMap
from tomato_recon.models.encoders.base import PointBackbone, masked_mean


class ResidualPointBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.block(value)


class PointNeXtAdapter(PointBackbone):
    """Portable PointNeXt-style baseline preserving the OpenPoints adapter contract.

    It intentionally uses core PyTorch operations so the default image is reproducible.
    A research deployment can replace the internal blocks with OpenPoints without changing
    inputs or outputs.
    """

    def __init__(self, input_dim: int = 6, output_dim: int = 96, global_dim: int = 128, **_: object) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.global_dim = global_dim
        self.input = nn.Sequential(
            nn.Linear(input_dim + 3, output_dim), nn.LayerNorm(output_dim), nn.GELU()
        )
        self.blocks = nn.Sequential(*[ResidualPointBlock(output_dim) for _ in range(3)])
        self.global_projection = nn.Sequential(
            nn.Linear(output_dim * 2, global_dim), nn.GELU(), nn.LayerNorm(global_dim)
        )

    def forward(
        self, xyz: Tensor, features: Tensor, mask: Tensor
    ) -> tuple[Tensor, list[FeatureMap], Tensor]:
        centre = masked_mean(xyz, mask, dim=1).unsqueeze(1)
        value = self.blocks(self.input(torch.cat([xyz - centre, features], dim=-1)))
        value = value * mask[..., None]
        mean = masked_mean(value, mask, dim=1)
        maximum = value.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)
        maximum = torch.nan_to_num(maximum, neginf=0.0)
        global_feature = self.global_projection(torch.cat([mean, maximum], dim=-1))
        return value, [FeatureMap(xyz=xyz, features=value, mask=mask)], global_feature

