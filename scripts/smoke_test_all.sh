#!/usr/bin/env bash
set -euo pipefail

python -m unittest discover -s tests -v
python -m tomato_recon.train.train_diffusion \
  trainer.fast_dev_run=true model.diffusion.variant=points \
  model.diffusion.sample_steps=4 model.diffusion.train_timesteps=20 \
  output.dir=outputs/diffts/smoke_points
python -m tomato_recon.train.train_diffusion \
  trainer.fast_dev_run=true model.diffusion.variant=points_stage1 \
  model.diffusion.sample_steps=4 model.diffusion.train_timesteps=20 \
  output.dir=outputs/diffts/smoke_points_stage1
