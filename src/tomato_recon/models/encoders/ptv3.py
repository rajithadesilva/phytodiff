from __future__ import annotations

import torch
from torch import Tensor, nn

from tomato_recon.data.schemas import FeatureMap
from tomato_recon.models.encoders.base import PointBackbone, masked_mean


class PTv3Adapter(PointBackbone):
    def __init__(self, output_dim: int = 192, global_dim: int = 256, **kwargs: object) -> None:
        super().__init__()
        try:
            from pointcept.models import build_model
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PTv3Adapter requires Pointcept and its compiled point operators."
            ) from exc
        self.output_dim = output_dim
        self.global_dim = global_dim
        pointcept_config = kwargs.get("pointcept_config")
        if not pointcept_config:
            raise ValueError(
                "PTv3Adapter requires model.encoder.pointcept_config matching the installed "
                "Pointcept release (including in_channels and encoder depths)."
            )
        self.grid_size = float(kwargs.get("grid_size", 0.01))
        self.model = build_model(pointcept_config)
        self.output_projection = nn.LazyLinear(output_dim)
        self.global_projection = nn.Sequential(
            nn.Linear(output_dim * 2, global_dim), nn.GELU(), nn.LayerNorm(global_dim)
        )

    def forward(self, xyz: Tensor, features: Tensor, mask: Tensor) -> tuple[Tensor, list[FeatureMap], Tensor]:
        counts = mask.sum(dim=1).long()
        batch_index = torch.repeat_interleave(
            torch.arange(len(xyz), device=xyz.device), counts
        )
        coordinate = xyz[mask]
        payload = {
            "coord": coordinate,
            "feat": features[mask],
            "batch": batch_index,
            "offset": counts.cumsum(0),
            "grid_coord": torch.floor(coordinate / self.grid_size).int(),
        }
        result = self.model(payload)
        sparse_features = result.feat if hasattr(result, "feat") else result["feat"]
        sparse_coordinate = result.coord if hasattr(result, "coord") else result.get("coord", coordinate)
        sparse_batch = result.batch if hasattr(result, "batch") else result.get("batch", batch_index)
        sparse_features = self.output_projection(sparse_features)
        flat = torch.empty((len(coordinate), self.output_dim), device=xyz.device, dtype=sparse_features.dtype)
        if len(sparse_features) == len(coordinate):
            flat[:] = sparse_features
        else:
            for batch in range(len(xyz)):
                query_mask = batch_index == batch
                source_mask = sparse_batch == batch
                if not source_mask.any():
                    raise RuntimeError(f"PTv3 returned no tokens for batch element {batch}")
                nearest = torch.cdist(
                    coordinate[query_mask][None], sparse_coordinate[source_mask][None]
                ).argmin(dim=-1).squeeze(0)
                flat[query_mask] = sparse_features[source_mask][nearest]
        dense = torch.zeros(
            (*xyz.shape[:2], self.output_dim), device=xyz.device, dtype=flat.dtype
        )
        dense[mask] = flat
        mean = masked_mean(dense, mask, dim=1)
        maximum = dense.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)
        maximum = torch.nan_to_num(maximum, neginf=0.0)
        global_feature = self.global_projection(torch.cat([mean, maximum], dim=-1))
        return dense, [FeatureMap(xyz=xyz, features=dense, mask=mask)], global_feature
