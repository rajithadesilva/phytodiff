# Tomato parametric reconstruction

This repository reconstructs one isolated tomato plant from a labelled or unlabelled 3D scan. The canonical output is a rooted, typed, attributed plant graph. A fixed-slot conditional diffusion model completes the centreline, a constrained arborescence decoder establishes biological topology, and differentiable PyTorch primitives generate editable stems, leaves, and confidence-supported optional fruit. Inference writes meshes, traits, uncertainty, and a metres/Z-up OpenUSD asset for NVIDIA Isaac Sim.

The implementation follows `papers/Tomato_Diffusion_Parametric_Reconstruction_Implementation_Spec.pdf`. TomatoWUR v3, TomatoPGT v1, and Pheno4D are external data: they are never downloaded by training code, copied into an image, or committed here.

## What is implemented

```text
source dataset adapters -> deterministic canonical fixed-K cache -> point encoder
    -> conditional skeleton diffusion -> typed nodes + sparse parent scores
    -> constrained directed spanning tree -> organ parameters
    -> differentiable geometry -> graph/PLY/traits/uncertainty/OpenUSD
```

- Typed contracts for samples, encoder output, skeleton predictions, graphs, organ parameters, geometry, and export reports.
- Source-specific adapters for TomatoWUR v3, complete TomatoPGT v1 scans, and annotated Pheno4D tomato scans. Every point cloud becomes one globally numbered instance in the same flat `data/dataset/` directory.
- Official TomatoWUR ground-truth skeleton preservation, TomatoPGT graph-path conversion, and deterministic Pheno4D centreline reconstruction from its manual stem/leaf instances.
- KPConvX is the default Stage 1 backbone, with PointNeXt and Sonata-PTv3 available as explicit alternatives with actionable dependency/checkpoint errors.
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

### 2. Mount the source datasets outside the image

Prerequisite: access to the datasets you intend to use and acceptance of their terms. The default configuration paths are `data/TomatoWUR`, `data/TomatoPGT_v1.0`, and `data/Pheno4D`.

```bash
test -d data/TomatoWUR/point_clouds
test -d data/TomatoWUR/ann_versions
test -d data/TomatoPGT_v1.0
test -d data/Pheno4D/Tomato01
```

Sources: [TomatoWUR v3](https://data.4tu.nl/datasets/e2c59841-4653-45de-a75e-4994b2766a2f/3), [TomatoPGT v1](https://data.mendeley.com/datasets/72md54c7n7/1), and [Pheno4D](https://www.ipb.uni-bonn.de/data/pheno4d/index.html). Verify that the corresponding file under `configs/data/` points at each mount. The code never opens an interactive downloader.

Format references: the [TomatoPGT paper](https://doi.org/10.1016/j.dib.2026.112642) and [CloudSeg/CloudGraph tools](https://github.com/nethpras/TomatoPGT-tools-binaries), plus the [Pheno4D paper](https://doi.org/10.1371/journal.pone.0256340) and [official data loaders](https://github.com/AIS-Bonn/data_loaders).

### 3. Run Stage 0 preprocessing and inspect fixed-K statistics

Prerequisite: Step 2. Run any subset of these commands; every adapter appends to the same globally numbered dataset.

```bash
docker compose -f docker/docker-compose.yml run --rm preprocess \
  python scripts/prepare_tomatowur.py --config configs/data/tomatowur_v3.yaml

docker compose -f docker/docker-compose.yml run --rm preprocess \
  python scripts/prepare_tomatopgt.py --config configs/data/tomatopgt_v1.yaml

docker compose -f docker/docker-compose.yml run --rm preprocess \
  python scripts/prepare_pheno4d.py --config configs/data/pheno4d_tomato.yaml

python -c "import json; m=json.load(open('data/dataset/manifest.json')); print({k: v['instance_count'] for k, v in m['datasets'].items()}, m['next_plant_number'])"
```

Expected: live progress shows discovery and the checking, processing, writing, completion, or skip status of every accepted point-cloud instance. `manifest.json` and one globally numbered `plant_<number>/` cache directory per point cloud contain versioned `.npz`, `.graph.json`, and `.params.json` files. There are no source-dataset subdirectories under `data/dataset/`. Existing source instances are skipped on reruns; new instances continue after the highest plant number already present.

The default source archives contribute:

- TomatoWUR: 44 instances with official split counts 35 train, 4 validation, and 5 test. Every official GT node and edge is preserved and padded to K=256.
- TomatoPGT: 42 complete PLY/annotation/graph instances. Four raw-only scans and `CH_07012025`, which has no graph, are ignored. The known `CH_07112025` annotation/`CH_07122025` graph filename mismatch is declared in the config. Original RGB and normals are retained; graph edge paths are deterministically resampled into K=256.
- Pheno4D: 77 annotated tomato instances with split counts 44 train, 11 validation, and 22 test. The 63 unlabelled tomato scans are ignored, maize directories are never discovered, unavailable RGB is represented by zeros, normals are estimated by local PCA, and centreline targets are reconstructed from the manual stem and temporally consistent leaf instances.

Running all three against an initially empty destination produces 163 complete instances. The converters reject cross-date split leakage by keeping every scan from the same physical source plant in one split.

#### Top-down partial point clouds

Every preparation command also writes `plant_<number>/top_down.npz` alongside the
full `sample.npz`. To add these clouds to an existing dataset without reading the
raw source scans, run from the repository root:

```bash
make generate-top-down
# Optional: change geometric occlusion strength.
make generate-top-down TOP_DOWN_OCCLUSION_RADIUS_M=0.002 TOP_DOWN_DEPTH_TOLERANCE_M=0.001
```

These targets run in the existing Docker Compose `preprocess` container. No host
Python environment or GPU is needed. The dataset and outputs are mounted from the
repository and remain available on the host. Set `DATASET_ROOT` to another path
under the mounted repository if needed. The equivalent direct Docker command is:

```bash
docker compose -f docker/docker-compose.yml run --rm \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 preprocess \
  python scripts/generate_top_down.py data/dataset \
  --occlusion-radius-m 0.001 --depth-tolerance-m 0.001
```

The view uses parallel downward rays along −Z in the canonical metres/Z-up frame.
Each valid source point contributes a circular XY footprint: a point is hidden
when another point within `occlusion_radius_m` is higher by more than
`depth_tolerance_m`. Increasing the radius increases occlusion; zero radius keeps
all valid points. Both values must be finite and non-negative and default to
0.001 metres. This is a sampled-surface approximation, so retained counts depend
on plant geometry and point density. No perspective, noise, or fixed-count
resampling is applied.

Override these settings during any source conversion using the existing dot-list
syntax, for example:

```bash
docker compose -f docker/docker-compose.yml run --rm preprocess \
  python scripts/prepare_pheno4d.py --config configs/data/pheno4d_tomato.yaml \
  top_down.occlusion_radius_m=0.002 top_down.depth_tolerance_m=0.001
```

Set `top_down.enabled=false` to disable automatic generation for that invocation;
existing partial files are retained. Top-down settings have their own generation
contract, so changing them regenerates partial files without rebuilding unchanged
full clouds or changing training compatibility. Conversion reruns also repair
missing or stale partial files when the full instance is otherwise skipped.

Each partial NPZ stores `xyz`, `rgb`, `normals`, `semantic`, `instance`, and
`point_valid` in source-row order, plus `source_point_indices` mapping to rows in
`sample.npz`. `metadata_json` preserves coordinate/provenance metadata and subsets
`point_to_original_index` when available. Its `top_down` section records the view,
settings, algorithm version, source checksum, and references to the full sample's
graph and parameter targets; target geometry stays in the existing files.
The manifest's per-instance `top_down` entry records the artifact path/checksum,
source checksum, settings, counts, and fraction of valid source points retained.
Training continues to load the full samples.

Matching artifacts are skipped. Missing, corrupt, or stale artifacts are replaced
atomically, and the standalone command reports per-instance progress and a JSON
summary. Individual failures are reported while other instances continue; any
failure gives the command a nonzero exit status.

### 4. Visualise the processed dataset

Prerequisite: at least one completed Stage 0 sample. `--count 0` renders every completed instance in the flat manifest; use a positive value for a smaller prefix.

```bash
make visualize-dataset
# Optional: preview a subset and adjust the viewing angle.
make visualize-dataset DATASET_VIS_COUNT=3 DATASET_VIS_AZIMUTH_DEG=35 DATASET_VIS_ELEVATION_DEG=15
```

Both visualization commands run in Docker. The default renders the entire dataset
to `outputs/dataset_preview`; override `DATASET_VIS_OUTPUT` to change the output
directory. The equivalent direct Docker command is:

```bash
docker compose -f docker/docker-compose.yml run --rm \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 preprocess \
  python scripts/visualize_dataset.py data/dataset --count 0 \
  --output outputs/dataset_preview

docker compose -f docker/docker-compose.yml run --rm preprocess \
  python -c "import json, pathlib; m=json.load(open('data/dataset/manifest.json')); expected=sum(x['status']=='complete' for x in m['instances']); actual=len(list(pathlib.Path('outputs/dataset_preview').glob('plant_*.png'))); print({'expected': expected, 'actual': actual}); assert actual == expected"
```

Expected: one six-panel PNG per completed point-cloud instance. With all three default sources this is 163 images. The top row keeps the front, side, and top views with cached target edges in red. Each viewport is framed using labelled plant points and skeleton nodes so background surfaces do not hide small plants. Pheno4D has no RGB, so its points use the canonical semantic-class palette.

The bottom row compares the full cloud, the saved top-down partial cloud, and an
occlusion overlay (green retained points, grey hidden points). All three share the
same mildly perspective camera and scale, viewed from the side at 15 degrees above
horizontal. Blue arrows show the downward sensor direction. Skeletons appear only
in the top row so they do not obscure the missing surfaces in the comparison.
Counts show actual valid/retained points before display subsampling; the header
includes the occlusion settings used to generate the saved cloud.

Use `--azimuth-deg 35 --elevation-deg 15` to adjust the comparison camera. These
options change only the visualization, not the sensor or saved clouds. Missing
top-down files show an unavailable message; stale files must be regenerated using
`scripts/generate_top_down.py` before visualizing them.

### 5. Train Stage 1: point encoder

Prerequisite: verified caches. Stage 1 exposes `pointnext`, `sonata_ptv3`, and
`kpconvx` (default). Prepare the pinned Sonata checkpoint once with
`make prepare-stage1-models`.

Train the default encoder on every training instance from all three sources:

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_encoder --config-name encoder \
  data.dataset=combined output.dir=outputs/encoder/combined
python -c "import torch; c=torch.load('outputs/encoder/combined/best.ckpt', map_location='cpu', weights_only=False); print(c['stage'], c['dataset_compatibility'], c['metrics'])"
```

Set `data.dataset` to a source ID to train on only that source while preserving its
train/validation/test split. Always use a separate output directory so one run cannot
overwrite another:

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_encoder --config configs/encoder/kpconvx.yaml \
  data.dataset=tomatowur output.dir=outputs/encoder/tomatowur

docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_encoder --config configs/encoder/kpconvx.yaml \
  data.dataset=tomatopgt output.dir=outputs/encoder/tomatopgt

docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_encoder --config configs/encoder/kpconvx.yaml \
  data.dataset=pheno4d output.dir=outputs/encoder/pheno4d
```

Valid selectors are `tomatowur`, `tomatopgt`, `pheno4d`, and `combined`. Selection is
performed from the source metadata in the shared flat manifest; no source-specific
subdirectories are created. Checkpoints record a deterministic compatibility signature
covering every selected source/preprocessing hash, so combined checkpoints no longer
depend on whichever plant happened to be sampled when they were saved.

Expected: `best.ckpt`, `last.ckpt`, resolved config, environment/git state, metrics,
and a cached validation prediction under the selected output directory. `best.ckpt` is
selected by the weighted validation overall score; test plants are never loaded by
training.

The former Stage 1 ablation is now the Stage 1 benchmark. It trains all three encoders
on each individual dataset and on the combined dataset: 12 independent training runs.
Each combined-trained checkpoint is then evaluated separately on TomatoWUR, TomatoPGT,
and Pheno4D, adding nine evaluation-only runs without retraining.

```bash
make stage1-benchmark
make stage1-benchmark STAGE1_BENCHMARK_RESUME=--resume
```

Completed `run_complete.json` entries are always skipped. Therefore, rerunning the first
command after completing the original 12-run benchmark performs only the new per-source
evaluations. Use the second command when an unfinished training run should resume from
its `last.ckpt`.

Results are written to `outputs/stage1_benchmark/<dataset>/<model>/`. The root
`comparison.json`, `comparison.csv`, and `comparison.md` report every validation and
held-out test metric, per-dataset rankings/winners, the combined-dataset winner, and
the cross-dataset mean validation ranking. They also report each combined-trained model
on each individual source; the corresponding raw metrics are stored under
`combined/<model>/by_dataset/<dataset>/`. `overall_scores.png` compares matched training
runs, `combined_models_by_dataset.png` compares the new per-source evaluations, and
`progress.jsonl` is the durable live-progress record. To run a subset directly, pass for
example `--datasets pheno4d combined --models kpconvx pointnext` to
`scripts/run_stage1_benchmark.py`.

To render the combined KPConvX benchmark test set again:

```bash
make visualize-stage1-test
```

Expected: six-panel prediction/target previews and test metrics in
`outputs/stage1_benchmark/combined/kpconvx/test_visualizations`. Do not use these test
results to tune training settings or thresholds.

### 6. Cache encoder features or predictions

Prerequisite: Stage 1 checkpoint and prediction file.

```bash
python scripts/cache_stage_predictions.py --stage encoder \
  --checkpoint outputs/stage1_benchmark/combined/kpconvx/best.ckpt \
  --input outputs/stage1_benchmark/combined/kpconvx/smoke_predictions.pt \
  --output outputs/cache/encoder
python -c "import json; print(json.load(open('outputs/cache/encoder/manifest.json')))"
```

Expected: a copied prediction cache plus checkpoint/prediction SHA-256 values. Verify the cache manifest before Stage 2.

### 7. Train Stage 2: fixed-K skeleton diffusion

Prerequisite: Stage 1 best checkpoint and matching preprocessing hash/K.

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_diffusion --config-name diffusion \
  data.dataset=combined \
  model.encoder.checkpoint=outputs/stage1_benchmark/combined/kpconvx/best.ckpt
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
  data.dataset=combined \
  model.encoder.checkpoint=outputs/stage1_benchmark/combined/kpconvx/best.ckpt \
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
  data.dataset=combined \
  model.encoder.checkpoint=outputs/stage1_benchmark/combined/kpconvx/best.ckpt \
  model.diffusion.checkpoint=outputs/diffusion/best.ckpt \
  model.graph.checkpoint=outputs/graph/best.ckpt \
  model.parametric.checkpoint=outputs/parametric/best.ckpt
python -c "import torch; c=torch.load('outputs/joint/best.ckpt', map_location='cpu', weights_only=False); print(c['upstream_checkpoint_hashes'])"
```

Expected: `outputs/joint/best.ckpt` containing the combined model and exact upstream hashes, plus `staged_validation.json` and `smoke_visibility_weights.pt`. Verify the frozen staged pass, occlusion-aware geometry, normal, skeleton, parameter, and radius losses and `discrete_decode_gradient = 0`.

### 12. Run frozen full test-set evaluation once

Prerequisite: a frozen experiment config and test predictions; do not tune on this output.

```bash
python -m tomato_recon.evaluate --processed-root data/dataset \
  --predictions outputs/inference --output outputs/evaluation/metrics.json
python -c "import json; m=json.load(open('outputs/evaluation/metrics.json')); print(m['primary_metric_groups'], m['aggregate'])"
```

Expected: per-sample and aggregate skeleton/topology/trait metrics. Geometry (including normal consistency when normals are available) and optional fruit are reported in `secondary_aggregate`.

### 13. Infer one sample and export all artifacts

Prerequisite: Stage 5 checkpoint and a processed `.npz` (or isolated TomatoWUR-style CSV/ASCII PLY).

```bash
python -m tomato_recon.infer --config-name infer \
  input.path=data/dataset/plant_000001/sample.npz \
  model.pipeline_checkpoint=outputs/joint/best.ckpt \
  inference.num_diffusion_samples=4 output.dir=outputs/inference/plant_000001
test -f outputs/inference/plant_000001/plant_graph.json && test -f outputs/inference/plant_000001/plant.usd
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
tar -czf outputs/combined_experiment_archive.tar.gz \
  configs outputs/stage1_benchmark outputs/encoder outputs/diffusion outputs/graph outputs/parametric \
  outputs/joint outputs/evaluation outputs/inference
tar -tzf outputs/combined_experiment_archive.tar.gz | head
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

All CLIs accept stable dot-list overrides after `--config-name` or `--config`. Every run records `resolved_config.yaml`, `environment.json`, and `git_state.json`. Checkpoints store model/optimizer state, stage, epoch, resolved config, label map, the selected dataset compatibility signature, K, git commit, metrics, and upstream checkpoint hashes. Loading fails on dataset/preprocessing-contract or K mismatches.

See [data documentation](docs/DATA.md), [training details](docs/TRAINING.md), [model contracts](docs/MODEL_CONTRACTS.md), and [USD export](docs/USD_EXPORT.md).

## Tests

Use the development image, which includes the pinned runtime dependencies and
test tools:

```bash
docker compose -f docker/docker-compose.yml run --rm train python -m pytest
```

For a local Python 3.11 environment, install the development extra first:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The suite covers data mapping/round trips/fixed-K reduction/hashes; spline, frames, stem, leaf, and fruit meshes; diffusion noising/loss/reverse sampling/pruning/determinism; candidate edges/constraints/arborescence; configuration/checkpoint compatibility; all five training scripts and resume; preprocessing-to-inference integration; and USD hierarchy/units/mesh metadata/reopen validation.
