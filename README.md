# Tomato parametric reconstruction

This repository reconstructs one isolated tomato plant from a labelled or unlabelled 3D scan. The canonical output is a rooted, typed, attributed plant graph. A fixed-slot conditional diffusion model completes the centreline, a constrained arborescence decoder establishes biological topology, and differentiable PyTorch primitives generate editable stems, leaves, and confidence-supported optional fruit. Inference writes meshes, traits, uncertainty, and a metres/Z-up OpenUSD asset for NVIDIA Isaac Sim.

The implementation follows `papers/Tomato_Diffusion_Parametric_Reconstruction_Implementation_Spec.pdf`. TomatoWUR v3 is external data: it is never downloaded by training code, copied into an image, or committed here.

## What is implemented

```text
TomatoWUR CSVs -> deterministic fixed-K cache -> point encoder
    -> conditional skeleton diffusion -> typed nodes + sparse parent scores
    -> constrained directed spanning tree -> organ parameters
    -> differentiable geometry -> graph/PLY/traits/uncertainty/OpenUSD
```

- Typed contracts for samples, encoder output, skeleton predictions, graphs, organ parameters, geometry, and export reports.
- Official TomatoWUR v3 CSV/JSON adapter, corrected ground-truth skeletons, metric root normalization, support-pole separation, deterministic voxel sampling, and lossless fixed-K padding.
- Configurable PointNeXt-style default backbone plus lazy Pointcept PTv3, Sonata-PTv3, and LitePT adapters with actionable dependency errors.
- Six-dimensional fixed-K diffusion over XYZ and parent flow, with existence/confidence heads, duplicate/bounds losses, FPS initialization, DDIM sampling, flow normalization, and NMS.
- Separate organ/role/visibility heads, sparse k-nearest parent scoring, hard botanical masks, root selection, maximum directed spanning arborescence, and main-stem continuity.
- Cubic spline, parallel-transport frames, tapered tube, leaf width-profile, and optional confidence-gated fruit geometry in PyTorch.
- Independent training stages, resumable checkpoint contracts, resolved configs, git/environment capture, smoke fixtures, evaluation, inference, USD/USDA export, and static/optional Isaac validation.

Fruit is a pseudo-labelled optional branch. It is confidence-gated and explicitly excluded from primary supervised metrics. The MVP does not implement plant dynamics, multi-plant row reconstruction, or future growth.

## Local smoke test

Python 3.11 is the validated interpreter. A CUDA GPU is optional for the tiny test fixture.

```bash
python -m pip install -e ".[dev]"
bash scripts/smoke_test_all.sh
```

The smoke configuration uses 64 points, K=16, and one forward/backward step per stage. It does not require TomatoWUR. Full settings live under `configs/`.

## Required ordered training and execution procedure

Run the following steps in this order. Each stage owns its checkpoint directory and can resume with `--resume PATH`.

### 1. Build the training image and verify GPU access

Prerequisite: Docker, BuildKit, NVIDIA Container Toolkit, and a compatible NVIDIA driver.

```bash
docker compose -f docker/docker-compose.yml build train
docker compose -f docker/docker-compose.yml run --rm train python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: the cached multi-stage image builds and the check prints `True` for CUDA on a GPU host. Verify the resolved image digest with `docker image inspect tomato-parametric-reconstruction-train --format '{{json .RepoDigests}}'` and archive it in Step 15.

### 2. Mount or download TomatoWUR v3 outside the image

Prerequisite: access to [TomatoWUR v3](https://data.4tu.nl/datasets/e2c59841-4653-45de-a75e-4994b2766a2f/3) and acceptance of its terms.

```bash
test -d data/raw/TomatoWUR_v3/point_clouds
test -d data/raw/TomatoWUR_v3/ann_versions
```

Expected: the external dataset has `point_clouds/`, `ann_versions/`, `images/`, and `camera_poses/`. Verify that `configs/data/tomatowur_v3.yaml` points at that mount. The code never opens an interactive downloader.

### 3. Run Stage 0 preprocessing and inspect fixed-K statistics

Prerequisite: Step 2 and plant-level split JSONs.

```bash
docker compose -f docker/docker-compose.yml run --rm preprocess \
  python scripts/prepare_tomatowur.py --config configs/data/tomatowur_v3.yaml
python -c "import json; m=json.load(open('data/processed/v3_gt_K256/manifest.json')); print(m['sample_count'], m['split_counts'], m['skeleton_annotation_version'], m['skeleton_modified_count'], m['warnings'])"
```

Expected: `manifest.json` and versioned `.npz`, `.graph.json`, and `.params.json` caches for all 44 plants. Verify the official plant-level counts are 35 train, 4 validation, and 5 test. Stage 0 reads `0-paper-2Dto3D_improved` and preserves every official GT node and parent edge exactly; it only translates coordinates to the root-centred frame and pads to K=256. It fails instead of modifying a skeleton if K is too small.

### 4. Visualise at least three processed plants

Prerequisite: at least three Stage 0 samples.

```bash
python scripts/visualize_dataset.py data/processed/v3_gt_K256 --count 3 --output outputs/dataset_preview
test "$(find outputs/dataset_preview -name '*.png' | wc -l)" -ge 3
```

Expected: three root-centred, three-view point-cloud/skeleton previews. Inspect root position, support-pole removal, labels, junctions, and tips. Red lines are the official GT parent edges; no repair edges are generated.

### 5. Train Stage 1: point encoder

Prerequisite: verified caches. Select a backbone with `model.encoder.name=pointnext|ptv3|sonata_ptv3|litept`; optional adapters require their external packages/configuration.

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_encoder --config-name encoder model.encoder.name=pointnext
python -c "import torch; c=torch.load('outputs/encoder/best.ckpt', map_location='cpu', weights_only=False); print(c['stage'], c['metrics'])"
```

Expected: `outputs/encoder/best.ckpt`, `outputs/encoder/last.ckpt`, resolved config, environment/git state, metrics, and a cached validation prediction. Every epoch trains only on the 35 training plants and evaluates all 4 validation plants without gradients. `best.ckpt` is selected by validation loss; test plants are never loaded by training.

After model selection is frozen, render all five test plants once:

```bash
make visualize-stage1-test
```

Expected: six-panel prediction/target previews and test metrics in `outputs/encoder/test_visualizations`. Do not use these test results to tune training settings or thresholds.

### 6. Cache encoder features or predictions

Prerequisite: Stage 1 checkpoint and prediction file.

```bash
python scripts/cache_stage_predictions.py --stage encoder \
  --checkpoint outputs/encoder/best.ckpt --input outputs/encoder/smoke_predictions.pt \
  --output outputs/cache/encoder
python -c "import json; print(json.load(open('outputs/cache/encoder/manifest.json')))"
```

Expected: a copied prediction cache plus checkpoint/prediction SHA-256 values. Verify the cache manifest before Stage 2.

### 7. Train Stage 2: fixed-K skeleton diffusion

Prerequisite: Stage 1 best checkpoint and matching preprocessing hash/K.

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_diffusion --config-name diffusion \
  model.encoder.checkpoint=outputs/encoder/best.ckpt
python -c "import torch; p=torch.load('outputs/diffusion/smoke_predictions.pt', weights_only=False); print(p['valid_mask'].sum(dim=1), p['confidence'].mean())"
```

Expected: `outputs/diffusion/best.ckpt` and sampled XYZ/flow/existence/confidence tensors. Verify retained-node counts, denoising metrics, normalized nonzero flow, and existence probabilities.

### 8. Cache diffusion predictions

Prerequisite: Stage 2 outputs.

```bash
python scripts/cache_stage_predictions.py --stage diffusion \
  --checkpoint outputs/diffusion/best.ckpt --input outputs/diffusion/smoke_predictions.pt \
  --output outputs/cache/diffusion
python -c "import json; print(json.load(open('outputs/cache/diffusion/manifest.json')))"
```

Expected: immutable train/validation diffusion predictions associated with one checkpoint hash. Do not mix predictions from incompatible K/preprocessing configurations.

### 9. Train Stage 3: biological graph

Prerequisite: encoder/diffusion checkpoints. The curriculum starts on ground-truth nodes, injects coordinate/flow noise, then accepts cached predicted-node experiments.

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_graph --config-name graph \
  model.encoder.checkpoint=outputs/encoder/best.ckpt \
  model.diffusion.checkpoint=outputs/diffusion/best.ckpt
python -c "import json; g=json.load(open('outputs/graph/smoke_graph.json')); print(g['root_node_id'], len(g['nodes']), len(g['edges']))"
```

Expected: `outputs/graph/best.ckpt` and a typed rooted graph. Verify one root, `edges = nodes - 1`, no support pole, no cycle, one component, and main-stem continuity.

### 10. Train Stage 4: parametric decoder

Prerequisite: Stage 3 checkpoint and automatically fitted parameter targets.

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_parametric --config-name parametric \
  model.graph.checkpoint=outputs/graph/best.ckpt
python -c "import json; p=json.load(open('outputs/parametric/smoke_parameters.json')); print(len(p['organs']))"
```

Expected: `outputs/parametric/best.ckpt`, `smoke_parameters.json`, and `smoke_geometry.ply`. Verify finite vertices/faces, positive radii, attachment IDs, and fitted stem/leaf parameters.

### 11. Run Stage 5 joint fine-tuning

Prerequisite: best checkpoints from Stages 1–4 with matching preprocessing hashes and K.

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_joint --config-name joint \
  model.encoder.checkpoint=outputs/encoder/best.ckpt \
  model.diffusion.checkpoint=outputs/diffusion/best.ckpt \
  model.graph.checkpoint=outputs/graph/best.ckpt \
  model.parametric.checkpoint=outputs/parametric/best.ckpt
python -c "import torch; c=torch.load('outputs/joint/best.ckpt', map_location='cpu', weights_only=False); print(c['upstream_checkpoint_hashes'])"
```

Expected: `outputs/joint/best.ckpt` containing the combined model and exact upstream hashes, plus `staged_validation.json` and `smoke_visibility_weights.pt`. Verify the frozen staged pass, occlusion-aware geometry, normal, skeleton, parameter, and radius losses and `discrete_decode_gradient = 0`.

### 12. Run frozen full test-set evaluation once

Prerequisite: a frozen experiment config and test predictions; do not tune on this output.

```bash
python -m tomato_recon.evaluate --processed-root data/processed/v3_gt_K256 \
  --predictions outputs/inference --output outputs/evaluation/metrics.json
python -c "import json; m=json.load(open('outputs/evaluation/metrics.json')); print(m['primary_metric_groups'], m['aggregate'])"
```

Expected: per-sample and aggregate skeleton/topology/trait metrics. Geometry (including normal consistency when normals are available) and optional fruit are reported in `secondary_aggregate`.

### 13. Infer one sample and export all artifacts

Prerequisite: Stage 5 checkpoint and a processed `.npz` (or isolated TomatoWUR-style CSV/ASCII PLY).

```bash
python -m tomato_recon.infer --config-name infer \
  input.path=data/processed/v3_gt_K256/samples/PLANT_ID.npz \
  model.pipeline_checkpoint=outputs/joint/best.ckpt \
  inference.num_diffusion_samples=4 output.dir=outputs/inference/PLANT_ID
test -f outputs/inference/PLANT_ID/plant_graph.json && test -f outputs/inference/PLANT_ID/plant.usd
```

Expected: the complete output contract described below, with no ground-truth node count used during inference.

### 14. Validate USD statically and optionally in Isaac Sim

Prerequisite: Step 13. Binary `.usd` requires the `usd-export` image/`usd` extra; without `pxr`, local export writes valid USDA text at the `.usd` path and reports the fallback.

```bash
docker compose -f docker/docker-compose.yml run --rm usd-export \
  python -m tomato_recon.export.isaac_validate outputs/inference/PLANT_ID/plant.usd \
  --report outputs/inference/PLANT_ID/static_validation.json
docker compose -f docker/docker-compose.yml --profile isaac-validate run --rm isaac-validate
```

Expected: a valid `/World/TomatoPlant`, metres-per-unit 1.0, Z-up, non-empty meshes, skeleton curves, materials, and machine-readable validation. Optional rendering unavailability does not fail schema validation.

### 15. Archive reproducibility artifacts

Prerequisite: the frozen run, metrics, and validation reports.

```bash
tar -czf outputs/tomatowur_experiment_archive.tar.gz \
  configs outputs/encoder outputs/diffusion outputs/graph outputs/parametric \
  outputs/joint outputs/evaluation outputs/inference
tar -tzf outputs/tomatowur_experiment_archive.tar.gz | head
```

Expected: resolved YAML, checkpoints and hashes, metrics, environment/git state, manifests/splits, per-sample predictions, USD reports, and the recorded container digest. Do not add the archive to git.

## Inference output contract

```text
outputs/inference/<plant_id>/
  input_normalised.ply       semantic_prediction.ply
  skeleton_prediction.ply   skeleton_prediction.json
  plant_graph.json           organ_parameters.json
  reconstructed_mesh.ply    traits.json
  uncertainty.json           plant.usd
  plant_debug.usda           export_report.json
  preview.png
```

## Configuration and checkpoints

All CLIs accept stable dot-list overrides after `--config-name` or `--config`. Every run records `resolved_config.yaml`, `environment.json`, and `git_state.json`. Checkpoints store model/optimizer state, stage, epoch, resolved config, label map, preprocessing hash, K, git commit, metrics, and upstream checkpoint hashes. Loading fails on preprocessing-hash or K mismatches.

See [data documentation](docs/DATA.md), [training details](docs/TRAINING.md), [model contracts](docs/MODEL_CONTRACTS.md), and [USD export](docs/USD_EXPORT.md).

## Tests

```bash
python -m pytest
# Standard-library fallback when pytest is not installed:
python -m unittest discover -s tests -v
```

The suite covers data mapping/round trips/fixed-K reduction/hashes; spline, frames, stem, leaf, and fruit meshes; diffusion noising/loss/reverse sampling/pruning/determinism; candidate edges/constraints/arborescence; configuration/checkpoint compatibility; all five training scripts and resume; preprocessing-to-inference integration; and USD hierarchy/units/mesh metadata/reopen validation.
