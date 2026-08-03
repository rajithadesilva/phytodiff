from __future__ import annotations

import torch
from torch import Tensor, nn


class DiffusionScheduler(nn.Module):
    def __init__(
        self,
        train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        if train_timesteps < 2:
            raise ValueError("train_timesteps must be at least two")
        betas = torch.linspace(beta_start, beta_end, train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.train_timesteps = train_timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    @staticmethod
    def _extract(values: Tensor, timestep: Tensor, target: Tensor) -> Tensor:
        result = values.gather(0, timestep.long())
        return result.view(timestep.shape[0], *([1] * (target.ndim - 1)))

    def q_sample(self, clean: Tensor, timestep: Tensor, noise: Tensor | None = None) -> Tensor:
        noise = torch.randn_like(clean) if noise is None else noise
        alpha_bar = self._extract(self.alpha_bar, timestep, clean)
        return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise

    def predict_clean(self, noisy: Tensor, predicted_noise: Tensor, timestep: Tensor) -> Tensor:
        alpha_bar = self._extract(self.alpha_bar, timestep, noisy)
        return (noisy - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt().clamp_min(1e-8)

    def ddim_step(
        self, predicted_noise: Tensor, timestep: Tensor, previous_timestep: int, noisy: Tensor
    ) -> Tensor:
        clean = self.predict_clean(noisy, predicted_noise, timestep)
        if previous_timestep < 0:
            return clean
        previous = self.alpha_bar[previous_timestep].to(noisy).view(1, *([1] * (noisy.ndim - 1)))
        return previous.sqrt() * clean + (1 - previous).sqrt() * predicted_noise

    def sampling_timesteps(self, steps: int) -> list[int]:
        if steps < 1:
            raise ValueError("sample_steps must be positive")
        return (
            torch.linspace(self.train_timesteps - 1, 0, min(steps, self.train_timesteps))
            .round()
            .long()
            .unique_consecutive()
            .tolist()
        )

