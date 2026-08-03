#!/usr/bin/env bash
set -euo pipefail

python -m tomato_recon.train.train_encoder --config configs/smoke/all.yaml
python -m tomato_recon.train.train_diffusion --config configs/smoke/all.yaml
python -m tomato_recon.train.train_graph --config configs/smoke/all.yaml
python -m tomato_recon.train.train_parametric --config configs/smoke/all.yaml
python -m tomato_recon.train.train_joint --config configs/smoke/all.yaml
python -m unittest discover -s tests -v

