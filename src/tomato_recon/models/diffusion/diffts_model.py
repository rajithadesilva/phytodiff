"""DiffTS sparse U-Net port using spconv.

Architecture and conditioning follow PRBonn/DiffTS@a1b75f5c4804ac018c553dcc6cf9a78bd518d8e4
(`DiffTS/models/diff_skeletonizer.py` and `DiffTS/models/models.py`).
Copyright (c) 2025 Photogrammetry & Robotics Bonn; MIT license.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

try:
    import spconv.pytorch as spconv
except ImportError:  # CPU-only host tooling can still inspect configs and data.
    spconv = None


def time_embedding(timestep: Tensor, width: int) -> Tensor:
    half = width // 2
    frequency = torch.exp(
        -math.log(10000) * torch.arange(half, device=timestep.device) / max(half - 1, 1)
    )
    phase = timestep.float()[:, None] * frequency[None]
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


def _activate(sparse, layer, norm):
    sparse = layer(sparse)
    return sparse.replace_feature(torch.nn.functional.leaky_relu(norm(sparse.features), 0.1))


class SparseResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, key: str) -> None:
        super().__init__()
        self.first = spconv.SubMConv3d(in_channels, out_channels, 3, padding=1, indice_key=key)
        self.second = spconv.SubMConv3d(out_channels, out_channels, 3, padding=1, indice_key=key)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.skip = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, sparse):
        hidden = _activate(sparse, self.first, self.norm1)
        hidden = self.second(hidden)
        return hidden.replace_feature(
            torch.nn.functional.leaky_relu(
                self.norm2(hidden.features) + self.skip(sparse.features), 0.1
            )
        )


class SparseDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, index: int) -> None:
        super().__init__()
        self.conv = spconv.SparseConv3d(
            in_channels, out_channels, 2, stride=2, indice_key=f"down{index}"
        )
        self.norm = nn.LayerNorm(out_channels)
        self.blocks = nn.Sequential(
            SparseResidual(out_channels, out_channels, f"subm{index}"),
            SparseResidual(out_channels, out_channels, f"subm{index}"),
        )

    def forward(self, sparse):
        return self.blocks(_activate(sparse, self.conv, self.norm))


class SparseUp(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, index: int) -> None:
        super().__init__()
        self.up = spconv.SparseInverseConv3d(
            in_channels, skip_channels, 2, indice_key=f"down{index}"
        )
        self.blocks = nn.Sequential(
            SparseResidual(skip_channels * 2, out_channels, f"subm{index-1}"),
            SparseResidual(out_channels, out_channels, f"subm{index-1}"),
        )

    def forward(self, sparse, skip):
        up = self.up(sparse)
        if not torch.equal(up.indices, skip.indices):
            raise RuntimeError("spconv inverse coordinates do not align with the DiffTS skip")
        return self.blocks(up.replace_feature(torch.cat((up.features, skip.features), -1)))


@dataclass
class SparseField:
    sparse: object
    inverse: Tensor
    origin_xyz: Tensor
    resolution: float


def sparse_field(xyz: Tensor, features: Tensor, resolution: float) -> SparseField:
    """Mean-quantize features and keep the inverse map to original points."""
    origin = xyz.amin(dim=0) - resolution
    grid_xyz = torch.round((xyz - origin) / resolution).int()
    unique, inverse, counts = torch.unique(
        grid_xyz, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    pooled = features.new_zeros((len(unique), features.shape[-1]))
    pooled.index_add_(0, inverse, features)
    pooled = pooled / counts[:, None].to(features.dtype)
    zyx = unique[:, [2, 1, 0]].int()
    indices = torch.cat((torch.zeros((len(zyx), 1), device=zyx.device, dtype=torch.int32), zyx), 1)
    shape = [max(16, int(v) + 17) for v in zyx.amax(0).tolist()]
    sparse = spconv.SparseConvTensor(pooled, indices, shape, 1)
    return SparseField(sparse, inverse, origin, resolution)


def sparse_xyz(sparse, origin: Tensor, resolution: float, stride: int = 1) -> Tensor:
    return origin + (sparse.indices[:, [3, 2, 1]].float() * stride + stride / 2) * resolution


@dataclass
class Condition:
    sparse: object
    origin_xyz: Tensor
    resolution: float
    count_logits: Tensor


class ConditionEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.stem = spconv.SubMConv3d(in_channels, channels[0], 3, padding=1, indice_key="csubm0")
        self.stem_norm = nn.LayerNorm(channels[0])
        self.stem_block = SparseResidual(channels[0], channels[0], "csubm0")
        self.down = nn.ModuleList([
            SparseDown(channels[i], channels[i + 1], i + 10) for i in range(4)
        ])

    def forward(self, field: SparseField):
        x = self.stem_block(_activate(field.sparse, self.stem, self.stem_norm))
        for down in self.down:
            x = down(x)
        return x


class LocalFusion(nn.Module):
    """DiffTS weighted-neighbour, latent, time, and multiplicative fusion."""

    def __init__(self, condition_channels: int, node_channels: int, time_channels: int) -> None:
        super().__init__()
        self.time_channels = time_channels
        self.neighbour_projection = nn.Linear(condition_channels, node_channels)
        self.latent = nn.Sequential(
            nn.Linear(node_channels, node_channels), nn.LeakyReLU(0.1),
            nn.Linear(node_channels, node_channels),
        )
        self.time = nn.Sequential(
            nn.Linear(time_channels, time_channels), nn.LeakyReLU(0.1),
            nn.Linear(time_channels, node_channels),
        )
        self.fuse = nn.Sequential(
            nn.Linear(node_channels * 2, node_channels), nn.LeakyReLU(0.1),
            nn.Linear(node_channels, node_channels),
        )

    def forward(self, sparse, node_xyz: Tensor, condition: Condition, timestep: Tensor):
        cond_xyz = sparse_xyz(condition.sparse, condition.origin_xyz, condition.resolution, 16)
        neighbours = min(8, len(cond_xyz))
        parts = []
        for start in range(0, len(node_xyz), 128):
            distance = torch.cdist(node_xyz[start:start + 128].float(), cond_xyz.float())
            values, indices = distance.topk(neighbours, dim=1, largest=False)
            weight = values.clamp_min(1e-6).reciprocal()
            weight = weight / weight.sum(1, keepdim=True)
            features = condition.sparse.features[indices]
            parts.append((features * weight[..., None]).sum(1))
        local = self.latent(self.neighbour_projection(torch.cat(parts, 0)))
        temp = self.time(time_embedding(timestep, self.time_channels)).expand(len(local), -1)
        weights = self.fuse(torch.cat((local, temp), -1))
        return sparse.replace_feature(sparse.features * weights)


class DiffTSModel(nn.Module):
    """Point-conditioned six-channel DiffTS with a target-free node-count head."""

    def __init__(
        self,
        variant: str,
        max_nodes: int = 256,
        *,
        condition_resolution: float = 0.001,
        node_resolution: float = 0.005,
        stage1_point_dim: int = 96,
        stage1_global_dim: int = 128,
        width_divisor: int = 2,
    ) -> None:
        super().__init__()
        if spconv is None:
            raise ImportError("DiffTS requires spconv; use the CUDA training image")
        if variant not in {"points", "points_stage1"}:
            raise ValueError("variant must be 'points' or 'points_stage1'")
        self.variant = variant
        self.max_nodes = max_nodes
        self.condition_resolution = condition_resolution
        self.node_resolution = node_resolution
        channels = tuple(v // width_divisor for v in (32, 32, 64, 128, 256))
        input_channels = 7 + (stage1_point_dim + stage1_global_dim if variant == "points_stage1" else 0)
        self.condition_encoder = ConditionEncoder(input_channels, channels)
        self.count_head = nn.Sequential(
            nn.Linear(channels[4], channels[4]), nn.LeakyReLU(0.1),
            nn.Linear(channels[4], max_nodes - 1),
        )
        self.stem = spconv.SubMConv3d(6, channels[0], 3, padding=1, indice_key="subm0")
        self.stem_norm = nn.LayerNorm(channels[0])
        self.stem_block = SparseResidual(channels[0], channels[0], "subm0")
        self.down = nn.ModuleList([
            SparseDown(channels[i], channels[i + 1], i + 1) for i in range(4)
        ])
        up_outputs = (channels[4], channels[3], channels[1] * 3, channels[0] * 3)
        up_widths = (channels[4], *up_outputs[:-1])
        self.up = nn.ModuleList([
            SparseUp(in_width, channels[3 - i], out_width, 4 - i)
            for i, (in_width, out_width) in enumerate(zip(up_widths, up_outputs))
        ])
        fusion_widths = list(channels[:4]) + list(up_widths)
        self.fusions = nn.ModuleList([
            LocalFusion(channels[4], width, channels[0] * 3) for width in fusion_widths
        ])
        self.out = nn.Sequential(
            nn.Linear(channels[0] * 3, channels[0] * 2), nn.LeakyReLU(0.1),
            nn.Linear(channels[0] * 2, 6),
        )

    def encode_condition(
        self, xyz: Tensor, rgb: Tensor, rgb_available: bool,
        stage1_points: Tensor | None = None, stage1_global: Tensor | None = None,
    ) -> Condition:
        flag = xyz.new_full((len(xyz), 1), float(rgb_available))
        input_features = [xyz, rgb if rgb_available else torch.zeros_like(rgb), flag]
        if self.variant == "points_stage1":
            if stage1_points is None or stage1_global is None:
                raise ValueError("points_stage1 requires both Stage 1 feature tensors")
            if len(stage1_points) != len(xyz):
                raise ValueError("Stage 1 features must align with conditioning points")
            input_features += [stage1_points, stage1_global.expand(len(xyz), -1)]
        elif stage1_points is not None or stage1_global is not None:
            raise ValueError("points variant must not consume Stage 1 features")
        field = sparse_field(xyz, torch.cat(input_features, -1), self.condition_resolution)
        encoded = self.condition_encoder(field)
        count_logits = self.count_head(encoded.features.mean(0, keepdim=True)).squeeze(0)
        return Condition(encoded, field.origin_xyz, field.resolution, count_logits)

    def denoise(self, state: Tensor, timestep: Tensor, condition: Condition) -> Tensor:
        field = sparse_field(state[:, :3], state, self.node_resolution)
        x = self.stem_block(_activate(field.sparse, self.stem, self.stem_norm))
        skips = [x]
        for index, down in enumerate(self.down):
            xyz = sparse_xyz(x, field.origin_xyz, field.resolution, 2 ** index)
            x = down(self.fusions[index](x, xyz, condition, timestep))
            skips.append(x)
        for index, up in enumerate(self.up):
            xyz = sparse_xyz(x, field.origin_xyz, field.resolution, 2 ** (4 - index))
            x = up(self.fusions[index + 4](x, xyz, condition, timestep), skips[3 - index])
        return self.out(x.features[field.inverse])
