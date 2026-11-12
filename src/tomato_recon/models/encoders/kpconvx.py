"""Dense KPConvX backbone adapted for the TomatoWUR Stage 1 contract.

The kernel-attention operation follows KPConvX (Thomas et al., CVPR 2024).  This
standalone implementation keeps neighbour construction in PyTorch so it can be
tested without Pointcept, while retaining the kernel-point modulation, grid
pyramid, skip decoder, and dense inverse mapping used by the official model.

Upstream reference: apple/ml-kpconvx@54e644a9f3bddd4c344a58193897a44582b0fea4
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from tomato_recon.data.schemas import FeatureMap
from tomato_recon.models.encoders.base import PointBackbone, masked_mean


def _kernel_points(shell_sizes: Sequence[int]) -> Tensor:
    """Create deterministic concentric kernel shells with a centre point."""
    points: list[list[float]] = []
    shell_count = len(shell_sizes)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for shell_index, count in enumerate(shell_sizes):
        if shell_index == 0 and count == 1:
            points.append([0.0, 0.0, 0.0])
            continue
        radius = shell_index / max(shell_count - 1, 1)
        for index in range(count):
            z = 1.0 - 2.0 * (index + 0.5) / count
            radial = math.sqrt(max(1.0 - z * z, 0.0))
            angle = index * golden_angle
            points.append(
                [radius * radial * math.cos(angle), radius * radial * math.sin(angle), radius * z]
            )
    return torch.tensor(points, dtype=torch.float32)


def _voxel_pool(points: Tensor, features: Tensor, voxel_size: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Mean-pool one cloud and return coordinates plus the dense inverse map."""
    grid = torch.floor(points / voxel_size).long()
    unique, inverse, counts = torch.unique(
        grid, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    pooled_points = points.new_zeros((len(unique), 3))
    pooled_features = features.new_zeros((len(unique), features.shape[-1]))
    pooled_points.index_add_(0, inverse, points)
    pooled_features.index_add_(0, inverse, features)
    scale = counts.to(points.dtype).unsqueeze(-1)
    return pooled_points / scale, pooled_features / scale, unique, inverse


@torch.no_grad()
def _grid_neighbors(
    points: Tensor,
    grid: Tensor,
    *,
    voxel_size: float,
    kernel_radius: float,
    limit: int,
    query_chunk: int = 4096,
) -> tuple[Tensor, Tensor]:
    """Find bounded neighbours through the occupied voxel hash table."""
    if not len(points):
        empty = torch.empty((0, limit), dtype=torch.long, device=points.device)
        return empty, points.new_empty((0, limit, 3))
    cell_radius = max(1, math.ceil(kernel_radius))
    offsets = torch.tensor(
        list(itertools.product(range(-cell_radius, cell_radius + 1), repeat=3)),
        dtype=grid.dtype,
        device=grid.device,
    )
    minimum = grid.amin(dim=0)
    shifted = grid - minimum
    dimensions = shifted.amax(dim=0) + 2 * cell_radius + 1
    shifted = shifted + cell_radius

    def ravel(value: Tensor) -> Tensor:
        return (value[..., 0] * dimensions[1] + value[..., 1]) * dimensions[2] + value[..., 2]

    hashes = ravel(shifted)
    sorted_hashes, order = hashes.sort()
    all_indices: list[Tensor] = []
    all_offsets: list[Tensor] = []
    radius_m = kernel_radius * voxel_size
    for start in range(0, len(points), query_chunk):
        stop = min(start + query_chunk, len(points))
        candidate_grid = shifted[start:stop, None, :] + offsets[None, :, :]
        inside = ((candidate_grid >= 0) & (candidate_grid < dimensions)).all(dim=-1)
        candidate_hash = ravel(candidate_grid.clamp_min(0))
        positions = torch.searchsorted(sorted_hashes, candidate_hash)
        positions_safe = positions.clamp_max(len(sorted_hashes) - 1)
        present = inside & (positions < len(sorted_hashes))
        present &= sorted_hashes[positions_safe] == candidate_hash
        candidate_index = order[positions_safe]
        query = points[start:stop, None, :]
        relative = points[candidate_index] - query
        distance = relative.square().sum(dim=-1).sqrt()
        distance = distance.masked_fill(~present | (distance > radius_m), torch.inf)
        take = min(limit, distance.shape[1])
        selected_distance, selected = distance.topk(take, dim=1, largest=False)
        selected_index = candidate_index.gather(1, selected)
        self_index = torch.arange(start, stop, device=points.device)[:, None]
        selected_index = torch.where(selected_distance.isfinite(), selected_index, self_index)
        if take < limit:
            selected_index = torch.cat(
                [selected_index, self_index.expand(-1, limit - take)], dim=1
            )
        all_indices.append(selected_index)
        all_offsets.append(points[selected_index] - query)
    return torch.cat(all_indices), torch.cat(all_offsets)


class KernelAttention(nn.Module):
    """KPConvX kernel modulation over a fixed local neighbourhood."""

    def __init__(
        self,
        channels: int,
        shell_sizes: Sequence[int],
        attention_groups: int,
        kernel_radius: float,
    ) -> None:
        super().__init__()
        if channels % attention_groups:
            raise ValueError("KPConvX channels must be divisible by attention_groups")
        kernels = _kernel_points(shell_sizes)
        self.register_buffer("kernel_points", kernels, persistent=True)
        self.groups = attention_groups
        self.channels_per_group = channels // attention_groups
        self.kernel_radius = kernel_radius
        self.modulation = nn.Linear(channels, len(kernels) * attention_groups)
        self.projection = nn.Linear(channels, channels)

    def forward(self, features: Tensor, neighbor_index: Tensor, relative: Tensor, radius_m: float) -> Tensor:
        normalized = relative / max(radius_m, 1e-8)
        kernel_distance = torch.cdist(normalized, self.kernel_points.to(normalized.dtype))
        nearest_distance, nearest_kernel = kernel_distance.min(dim=-1)
        influence = (1.0 - nearest_distance).clamp_min(0.0)
        modulation = self.modulation(features).view(
            len(features), len(self.kernel_points), self.groups
        ).sigmoid()
        modulation = modulation.gather(
            1, nearest_kernel[..., None].expand(-1, -1, self.groups)
        )
        weights = modulation * influence[..., None]
        neighbors = features[neighbor_index].view(
            len(features), neighbor_index.shape[1], self.groups, self.channels_per_group
        )
        aggregated = (neighbors * weights[..., None]).sum(dim=1)
        aggregated = aggregated / weights.sum(dim=1).clamp_min(1e-6)[..., None]
        return self.projection(aggregated.flatten(1))


class KPConvXBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        shell_sizes: Sequence[int],
        attention_groups: int,
        kernel_radius: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.kernel = KernelAttention(channels, shell_sizes, attention_groups, kernel_radius)
        self.mlp = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, features: Tensor, neighbor_index: Tensor, relative: Tensor, radius_m: float) -> Tensor:
        value = features + self.kernel(self.norm(features), neighbor_index, relative, radius_m)
        return value + self.mlp(value)


class KPConvXAdapter(PointBackbone):
    """Memory-conscious KPConvX encoder/decoder with dense per-point output."""

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 96,
        global_dim: int = 128,
        grid_size: float = 0.005,
        shell_sizes: Sequence[int] = (1, 14, 28),
        layer_blocks: Sequence[int] = (2, 2, 2, 8, 2),
        init_channels: int = 32,
        channel_scaling: float = math.sqrt(2.0),
        radius_scaling: float = 2.2,
        neighbor_limits: Sequence[int] = (12, 16, 20, 20, 20),
        kernel_radius: float = 2.3,
        attention_groups: int = 8,
        **_: object,
    ) -> None:
        super().__init__()
        if len(layer_blocks) != len(neighbor_limits):
            raise ValueError("KPConvX layer_blocks and neighbor_limits must have equal length")
        self.output_dim = output_dim
        self.global_dim = global_dim
        self.grid_size = float(grid_size)
        self.radius_scaling = float(radius_scaling)
        self.kernel_radius = float(kernel_radius)
        self.neighbor_limits = tuple(int(value) for value in neighbor_limits)
        channels = [
            max(attention_groups, math.ceil(init_channels * channel_scaling**level / 16) * 16)
            for level in range(len(layer_blocks))
        ]
        self.channels = channels
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, channels[0]), nn.LayerNorm(channels[0]), nn.GELU()
        )
        self.down_projections = nn.ModuleList(
            nn.Linear(channels[level - 1], channels[level])
            for level in range(1, len(channels))
        )
        self.encoder = nn.ModuleList(
            nn.ModuleList(
                KPConvXBlock(width, shell_sizes, attention_groups, kernel_radius)
                for _ in range(int(depth))
            )
            for width, depth in zip(channels, layer_blocks)
        )
        self.decoder_projection = nn.ModuleList(
            nn.Sequential(
                nn.Linear(channels[level] + channels[level + 1], channels[level]),
                nn.LayerNorm(channels[level]),
                nn.GELU(),
            )
            for level in range(len(channels) - 1)
        )
        self.decoder_blocks = nn.ModuleList(
            KPConvXBlock(channels[level], shell_sizes, attention_groups, kernel_radius)
            for level in range(len(channels) - 1)
        )
        self.output_projection = nn.Linear(channels[0], output_dim)
        self.global_projection = nn.Sequential(
            nn.Linear(output_dim * 2, global_dim), nn.GELU(), nn.LayerNorm(global_dim)
        )

    def _cloud(self, points: Tensor, features: Tensor) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        level_points: list[Tensor] = []
        level_features: list[Tensor] = []
        level_grids: list[Tensor] = []
        inverse_maps: list[Tensor] = []

        points0, features0, grid0, dense_inverse = _voxel_pool(
            points, features, self.grid_size
        )
        value = self.input_projection(features0)
        current_points, current_grid = points0, grid0
        for level, blocks in enumerate(self.encoder):
            if level:
                size = self.grid_size * self.radius_scaling**level
                current_points, value, current_grid, inverse = _voxel_pool(
                    current_points, value, size
                )
                inverse_maps.append(inverse)
                value = self.down_projections[level - 1](value)
            size = self.grid_size * self.radius_scaling**level
            neighbors, relative = _grid_neighbors(
                current_points,
                current_grid,
                voxel_size=size,
                kernel_radius=self.kernel_radius,
                limit=self.neighbor_limits[level],
            )
            for block in blocks:
                value = block(value, neighbors, relative, self.kernel_radius * size)
            level_points.append(current_points)
            level_features.append(value)
            level_grids.append(current_grid)

        decoded = level_features[-1]
        for level in reversed(range(len(level_features) - 1)):
            decoded = decoded[inverse_maps[level]]
            decoded = self.decoder_projection[level](
                torch.cat([level_features[level], decoded], dim=-1)
            )
            size = self.grid_size * self.radius_scaling**level
            neighbors, relative = _grid_neighbors(
                level_points[level],
                level_grids[level],
                voxel_size=size,
                kernel_radius=self.kernel_radius,
                limit=self.neighbor_limits[level],
            )
            decoded = self.decoder_blocks[level](
                decoded, neighbors, relative, self.kernel_radius * size
            )

        dense = self.output_projection(decoded)[dense_inverse]
        pyramid = list(zip(level_points, level_features))
        return dense, pyramid

    @staticmethod
    def _padded_map(clouds: list[tuple[Tensor, Tensor]]) -> FeatureMap:
        batch = len(clouds)
        maximum = max(len(points) for points, _ in clouds)
        width = clouds[0][1].shape[-1]
        xyz = clouds[0][0].new_zeros((batch, maximum, 3))
        features = clouds[0][1].new_zeros((batch, maximum, width))
        mask = torch.zeros((batch, maximum), dtype=torch.bool, device=xyz.device)
        for index, (points, values) in enumerate(clouds):
            xyz[index, : len(points)] = points
            features[index, : len(values)] = values
            mask[index, : len(points)] = True
        return FeatureMap(xyz=xyz, features=features, mask=mask)

    def forward(
        self, xyz: Tensor, features: Tensor, mask: Tensor
    ) -> tuple[Tensor, list[FeatureMap], Tensor]:
        dense = features.new_zeros((*features.shape[:2], self.output_dim))
        per_level: list[list[tuple[Tensor, Tensor]]] = [
            [] for _ in range(len(self.channels))
        ]
        for batch_index in range(len(xyz)):
            valid = mask[batch_index]
            if not valid.any():
                continue
            cloud_dense, cloud_pyramid = self._cloud(
                xyz[batch_index, valid], features[batch_index, valid]
            )
            dense[batch_index, valid] = cloud_dense
            for level, cloud in enumerate(cloud_pyramid):
                per_level[level].append(cloud)
        pyramid = [self._padded_map(level) for level in per_level if len(level) == len(xyz)]
        mean = masked_mean(dense, mask, dim=1)
        maximum = dense.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)
        maximum = torch.nan_to_num(maximum, neginf=0.0)
        global_feature = self.global_projection(torch.cat([mean, maximum], dim=-1))
        return dense, pyramid, global_feature
