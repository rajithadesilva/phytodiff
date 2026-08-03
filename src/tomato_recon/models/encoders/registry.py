"""Lazy point-backbone registry with actionable optional-dependency failures."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig, OmegaConf

from tomato_recon.models.encoders.base import PointBackbone

_REGISTRY: dict[str, tuple[str, str, str | None, str | None]] = {
    "pointnext": ("tomato_recon.models.encoders.pointnext", "PointNeXtAdapter", None, None),
    "ptv3": (
        "tomato_recon.models.encoders.ptv3",
        "PTv3Adapter",
        "pointcept",
        "Install Pointcept and its CUDA point operators, then reinstall this package.",
    ),
    "sonata_ptv3": (
        "tomato_recon.models.encoders.sonata",
        "SonataPTv3Adapter",
        "pointcept",
        "Install Pointcept/Sonata and provide an official compatible checkpoint.",
    ),
    "litept": (
        "tomato_recon.models.encoders.litept",
        "LitePTAdapter",
        "litept",
        "Install the optional LitePT dependency before selecting model.encoder.name=litept.",
    ),
}


def register_backbone(name: str) -> Callable[[type[PointBackbone]], type[PointBackbone]]:
    def decorator(cls: type[PointBackbone]) -> type[PointBackbone]:
        _REGISTRY[name] = (cls.__module__, cls.__name__, None, None)
        return cls

    return decorator


def list_backbones() -> dict[str, bool]:
    return {name: dependency is None or importlib.util.find_spec(dependency) is not None for name, (*_, dependency, _) in _REGISTRY.items()}


def ensure_backbone_available(name: str) -> None:
    if name not in _REGISTRY:
        raise ValueError(f"unknown point backbone {name!r}; choose one of {sorted(_REGISTRY)}")
    _, _, dependency, hint = _REGISTRY[name]
    if dependency is not None and importlib.util.find_spec(dependency) is None:
        raise ImportError(f"backbone {name!r} requires optional package {dependency!r}. {hint}")


def create_backbone(name: str, **kwargs: Any) -> PointBackbone:
    ensure_backbone_available(name)
    module_name, class_name, _, _ = _REGISTRY[name]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def create_backbone_from_config(config: DictConfig) -> PointBackbone:
    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("model.encoder configuration must be a mapping")
    name = str(values.pop("name"))
    values.pop("num_semantic_classes", None)
    return create_backbone(name, **values)
