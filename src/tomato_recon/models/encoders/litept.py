from __future__ import annotations

from torch import Tensor

from tomato_recon.data.schemas import FeatureMap
from tomato_recon.models.encoders.base import PointBackbone


class LitePTAdapter(PointBackbone):
    def __init__(self, output_dim: int = 192, global_dim: int = 256, **kwargs: object) -> None:
        super().__init__()
        try:
            import litept
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("LitePTAdapter requires the optional litept package.") from exc
        self.output_dim = output_dim
        self.global_dim = global_dim
        self.model = litept.build_model(**kwargs)

    def forward(self, xyz: Tensor, features: Tensor, mask: Tensor) -> tuple[Tensor, list[FeatureMap], Tensor]:
        result = self.model(xyz=xyz, features=features, mask=mask)
        return result.point_features, result.multiscale_features, result.global_feature

