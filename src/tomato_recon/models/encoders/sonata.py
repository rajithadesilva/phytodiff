from __future__ import annotations

from pathlib import Path

import torch

from tomato_recon.models.encoders.ptv3 import PTv3Adapter


class SonataPTv3Adapter(PTv3Adapter):
    def __init__(self, checkpoint: str | None = None, tune_mode: str = "linear_probe", **kwargs: object) -> None:
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                "Sonata-PTv3 requires an official local checkpoint; set model.encoder.checkpoint."
            )
        super().__init__(**kwargs)
        self.tune_mode = tune_mode
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload.get("model_state", payload))
        cleaned = {
            key.removeprefix("module.").removeprefix("model."): value
            for key, value in state.items()
        }
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if len(missing) == len(self.model.state_dict()):
            raise ValueError("Sonata checkpoint does not contain compatible PTv3 model weights")
        if tune_mode in {"freeze", "linear_probe"}:
            self.model.requires_grad_(False)
        elif tune_mode != "full_finetune":
            raise ValueError("Sonata tune_mode must be freeze, linear_probe, or full_finetune")
