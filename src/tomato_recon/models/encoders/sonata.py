"""Official Sonata PTv3 backbone adapted to dense TomatoWUR predictions."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from tomato_recon.data.schemas import FeatureMap
from tomato_recon.models.encoders.base import PointBackbone, masked_mean
from tomato_recon.models.encoders.kpconvx import _voxel_pool
from tomato_recon.models.pretrained import verify_sonata_checkpoint


class SonataPTv3Adapter(PointBackbone):
    def __init__(
        self,
        pretrained_checkpoint: str | None = None,
        tune_mode: str = "linear_probe",
        input_dim: int = 6,
        output_dim: int = 96,
        global_dim: int = 128,
        grid_size: float = 0.005,
        patch_size: int = 128,
        enable_flash: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if input_dim != 6:
            raise ValueError("Sonata expects RGB and normal features (input_dim=6)")
        if pretrained_checkpoint is None:
            raise FileNotFoundError(
                "Sonata requires model.encoder.pretrained_checkpoint; "
                "run `make prepare-stage1-models`."
            )
        checkpoint = Path(pretrained_checkpoint)
        verify_sonata_checkpoint(checkpoint)
        try:
            import sonata
        except (ImportError, OSError) as exc:  # pragma: no cover - Docker dependency
            raise ImportError(
                "Sonata dependencies are unavailable; rebuild the Docker training image."
            ) from exc

        self.output_dim = output_dim
        self.global_dim = global_dim
        self.input_dim = input_dim
        self.grid_size = float(grid_size)
        self.tune_mode = tune_mode
        custom_config = {
            "enable_flash": bool(enable_flash),
            "enc_patch_size": [int(patch_size)] * 5,
            "freeze_encoder": tune_mode in {"freeze", "linear_probe"},
        }
        self.model = sonata.model.load(str(checkpoint), custom_config=custom_config)
        # The official checkpoint was trained with Collect(feat_keys=("coord",
        # "color", "normal")): three centred coordinate values followed by RGB
        # and normals.  Keep the Stage 1 interface as RGB + normals, and add the
        # official coordinate features below after applying Sonata's CenterShift.
        self.expected_input_dim = int(self.model.embedding.in_channels)
        if self.expected_input_dim not in {input_dim, input_dim + 3}:
            raise ValueError(
                "unsupported Sonata checkpoint input dimension "
                f"{self.expected_input_dim}; expected {input_dim} or {input_dim + 3}"
            )
        if tune_mode in {"freeze", "linear_probe"}:
            self.model.requires_grad_(False)
            self.model.eval()
        elif tune_mode != "full_finetune":
            raise ValueError("Sonata tune_mode must be freeze, linear_probe, or full_finetune")
        self.output_projection = nn.LazyLinear(output_dim)
        self.global_projection = nn.Sequential(
            nn.Linear(output_dim * 2, global_dim), nn.GELU(), nn.LayerNorm(global_dim)
        )

    def train(self, mode: bool = True) -> "SonataPTv3Adapter":
        super().train(mode)
        if self.tune_mode in {"freeze", "linear_probe"}:
            self.model.eval()
        return self

    @staticmethod
    def _restore_input_resolution(point: object) -> object:
        # Sonata's official feature extraction recipe concatenates the first two
        # skip levels, then propagates the remaining parent features.
        for _ in range(2):
            if "pooling_parent" not in point.keys():
                break
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        return point

    def forward(
        self, xyz: Tensor, features: Tensor, mask: Tensor
    ) -> tuple[Tensor, list[FeatureMap], Tensor]:
        coordinates: list[Tensor] = []
        values: list[Tensor] = []
        grids: list[Tensor] = []
        inverses: list[Tensor] = []
        counts: list[int] = []
        for batch_index in range(len(xyz)):
            valid = mask[batch_index]
            if not valid.any():
                counts.append(0)
                inverses.append(torch.empty(0, dtype=torch.long, device=xyz.device))
                continue
            valid_xyz = xyz[batch_index, valid]
            # Match Sonata's default CenterShift(apply_z=True) transform.  Besides
            # making its coordinate feature compatible with the pretrained stem,
            # this keeps grid coordinates non-negative after the per-cloud shift.
            lower = valid_xyz.amin(dim=0)
            upper = valid_xyz.amax(dim=0)
            shift = torch.stack(
                ((lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, lower[2])
            )
            sonata_xyz = valid_xyz - shift
            pooled_xyz, pooled_features, grid, inverse = _voxel_pool(
                sonata_xyz, features[batch_index, valid], self.grid_size
            )
            # Sonata's GridSample subtracts the minimum grid coordinate before
            # serialization; Morton/Hilbert encoding requires non-negative cells.
            grid = grid - grid.amin(dim=0, keepdim=True)
            coordinates.append(pooled_xyz)
            if self.expected_input_dim == self.input_dim + 3:
                values.append(torch.cat([pooled_xyz, pooled_features], dim=-1))
            else:
                values.append(pooled_features)
            grids.append(grid)
            inverses.append(inverse)
            counts.append(len(pooled_xyz))
        if not coordinates:
            dense = features.new_zeros((*features.shape[:2], self.output_dim))
            global_feature = features.new_zeros((len(xyz), self.global_dim))
            return dense, [FeatureMap(xyz=xyz, features=dense, mask=mask)], global_feature

        coordinate = torch.cat(coordinates)
        batch_index = torch.repeat_interleave(
            torch.arange(len(counts), device=xyz.device),
            torch.tensor(counts, device=xyz.device),
        )
        payload = {
            "coord": coordinate,
            "grid_coord": torch.cat(grids).int(),
            "feat": torch.cat(values),
            "batch": batch_index,
            "offset": torch.tensor(counts, device=xyz.device).cumsum(0),
        }
        point = self._restore_input_resolution(self.model(payload))
        sparse_features = self.output_projection(point.feat)
        dense = sparse_features.new_zeros((*features.shape[:2], self.output_dim))
        start = 0
        for index, (count, inverse) in enumerate(zip(counts, inverses)):
            valid = mask[index]
            if count:
                dense[index, valid] = sparse_features[start : start + count][inverse]
            start += count
        mean = masked_mean(dense, mask, dim=1)
        maximum = dense.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)
        maximum = torch.nan_to_num(maximum, neginf=0.0)
        global_feature = self.global_projection(torch.cat([mean, maximum], dim=-1))
        return dense, [FeatureMap(xyz=xyz, features=dense, mask=mask)], global_feature
