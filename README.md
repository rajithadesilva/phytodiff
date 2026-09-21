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

Pheno4D uses its raw +Z as the biological up direction: soil lies in XY and the
stem grows above it. The adapter converts millimetres to metres and centres the
plant at its reconstructed stem base. Its `raw-z-up-v2` coordinate contract
corrects the retired viewer-oriented mapping, which incorrectly put height into
Y. Relative to that old cache, the rotation is `(X, Y, Z) -> (X, -Z, Y)` before
root centring. Skeletons, normals, parameter targets, and top-down clouds are
computed in the corrected frame, and `normalised_to_original` maps back to raw
millimetres. The frame version participates in preprocessing hashes, so old
Pheno4D caches are rebuilt automatically. Checkpoints trained on the old Pheno4D
or combined dataset require retraining; their compatibility signatures no longer
match the corrected dataset.

The released `T02_0325_a` annotation swaps soil and stem labels. The converter
maps source `0 -> 1` and `1 -> 0` for that scan before producing semantic and
instance targets; leaf IDs and the raw file are preserved. This correction is
recorded in `source_label_correction` metadata and the preprocessing contract.

#### Fixed-view partial point clouds

Every preparation command writes `plant_<number>/top_down.npz` and
`plant_<number>/side.npz` alongside the full `full.npz`. The side sensor is at
+Y and looks along −Y onto X–Z. To add either view to an existing dataset without
reading the raw source scans, run from the repository root:

```bash
make generate-top-down
make generate-side
# Optional: change geometric occlusion strength.
make generate-top-down TOP_DOWN_OCCLUSION_RADIUS_M=0.002 TOP_DOWN_DEPTH_TOLERANCE_M=0.001
make generate-side SIDE_OCCLUSION_RADIUS_M=0.002 SIDE_DEPTH_TOLERANCE_M=0.001
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

docker compose -f docker/docker-compose.yml run --rm \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 preprocess \
  python scripts/generate_side.py data/dataset \
  --occlusion-radius-m 0.001 --depth-tolerance-m 0.001
```

Top-down uses parallel rays along −Z with XY footprints. Side uses parallel rays
along −Y with XZ footprints; a point is hidden when a nearby point has greater Y
by more than `depth_tolerance_m`. Increasing either view's radius increases
occlusion; zero radius keeps all valid points. Both values must be finite and
non-negative and default to 0.001 metres. This is a sampled-surface approximation,
so retained counts depend on plant geometry and density. No perspective, noise,
or fixed-count resampling is applied.

Override these settings during any source conversion using the existing dot-list
syntax, for example:

```bash
docker compose -f docker/docker-compose.yml run --rm preprocess \
  python scripts/prepare_pheno4d.py --config configs/data/pheno4d_tomato.yaml \
  top_down.occlusion_radius_m=0.002 side.occlusion_radius_m=0.002
```

Set `top_down.enabled=false` or `side.enabled=false` to disable that automatic
artifact for an invocation; existing partial files are retained. Each view has an
independent generation contract, so changing its settings regenerates only that
partial file without rebuilding the full cloud or changing training compatibility.
Conversion reruns repair missing, corrupt, or stale partial files even when the
full instance is otherwise skipped.

Each partial NPZ stores `xyz`, `rgb`, `normals`, `semantic`, `instance`, and
`point_valid` in source-row order, plus `source_point_indices` mapping to rows in
`full.npz`. `metadata_json` preserves coordinate/provenance metadata and subsets
`point_to_original_index` when available. Its `top_down` or `side` section records
the view direction, projection, settings, algorithm version, source checksum, and
references to the full sample's graph and parameter targets; target geometry stays
in the existing files. The matching per-instance manifest entry records the
artifact path/checksum, source checksum, settings, counts, and retained fraction.

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

Expected: one nine-panel PNG per completed point-cloud instance. With all three default sources this is 163 images. The top row keeps the front, side, and top views with cached target edges in red. Each viewport is framed using labelled plant points and skeleton nodes so background surfaces do not hide small plants. Pheno4D has no RGB, so its points use the canonical semantic-class palette.

The second row compares full, top-down, and top-down occlusion. The third row does
the same for side. Each row shares a mildly offset perspective camera and scale.
Blue arrows show the corresponding −Z or −Y sensor direction. Skeletons appear
only in the top row so they do not obscure missing surfaces in the comparisons.
Counts show actual valid/retained points before display subsampling; the header
includes the occlusion settings used to generate the saved cloud.

Use `--azimuth-deg 35 --elevation-deg 15` to adjust the comparison camera. These
options change only the visualization, not the sensor or saved clouds. Missing
derived files show an unavailable message; stale files must be regenerated with
`make generate-top-down` or `make generate-side` before visualizing them.

### 5. Train Stage 1: point encoder

Prerequisite: verified caches. Stage 1 exposes `pointnext`, `sonata_ptv3`, and
`kpconvx` (default). Prepare the pinned Sonata checkpoint once with
`make prepare-stage1-models`.

Train the default encoder on every training instance from all three sources:

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_encoder --config-name encoder \
  data.dataset=combined data.pcl_types=[full] output.dir=outputs/encoder/combined
python -c "import torch; c=torch.load('outputs/encoder/combined/best.ckpt', map_location='cpu', weights_only=False); print(c['stage'], c['dataset_compatibility'], c['metrics'])"
```

`data.pcl_types` is the only view-selection interface for every encoder and later
training stage. It is a non-empty ordered array containing unique entries from
`full`, `top_down`, and `side`:

- `[full]` uses each `full.npz` once (the default).
- `[side]` uses each `side.npz` once.
- `[full,top_down,side]` presents three examples per plant in that order, with
  every derived view sharing the full reconstruction target.

Training, validation, and test metrics aggregate every selected view and reports
record the ordered array plus per-view sample counts. Derived views require current
artifacts. Use a separate output directory for each selection. The shared loader
applies the array to PointNeXt, Sonata PTv3, KPConvX, and downstream stages.

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

Stage 1 has two complementary ablations. Ablation 1 compares all three encoder
architectures on one source-dataset selection and one ordered point-cloud configuration.
The default uses all sources and the full point cloud:

```bash
make stage1-ablation-1
make stage1-ablation-1 STAGE1_ABLATION1_RESUME=--resume
make stage1-ablation-1 STAGE1_ABLATION1_PCL_TYPES="full top_down side" \
  STAGE1_ABLATION1_OUTPUT=outputs/stage1_ablation_1_all_views
```

Results are written to `outputs/stage1_ablation_1/<dataset>/<model>/`. The root
comparison files report validation rankings and held-out test metrics. `winner.json`
records the validation winner and is the input contract for Ablation 2. Run a subset
directly with `--models`, or choose a source using `--dataset`:

```bash
python scripts/run_stage1_ablation_1.py --dataset pheno4d \
  --pcl-types full side --models kpconvx pointnext
```

Ablation 2 reads the Ablation 1 winner, retrains that architecture on every nonempty
combination of `full`, `top_down`, and `side`, and evaluates all seven checkpoints on
all seven held-out test configurations. This produces 49 test cells without using test
results for model or epoch selection:

```bash
make stage1-ablation-2
make stage1-ablation-2 STAGE1_ABLATION2_RESUME=--resume
```

Outputs are written under `outputs/stage1_ablation_2/`. `matrix.json`, `matrix.csv`, and
`matrix.md` contain the complete results. Five `matrix_*.png` heatmaps cover semantic
mIoU, skeleton F1, centreline-offset score, junction F1, and weighted overall score.
Rows are training configurations and columns are test configurations. Pair and triple
configurations pool their selected views as separate samples. Both ablations run in the
Docker training service and require a visible CUDA GPU.

Ablation 3 measures cross-dataset generalization with the default KPConvX encoder. It
trains four checkpoints using `full`, `top_down`, and `side` as separate samples: one
checkpoint for each source dataset and one for their combined training split. Every
checkpoint is then evaluated on the held-out test split for all four dataset selections,
producing a 4-by-4 matrix. Validation from the training dataset selects each best epoch;
test results are never used for selection:

```bash
make stage1-ablation-3
make stage1-ablation-3 STAGE1_ABLATION3_RESUME=--resume
make stage1-ablation-3 STAGE1_ABLATION3_BATCH_SIZE=2
```

The training batch size defaults to `1`, matching the other Stage 1 ablations. Larger
batches are supported, but mixed view batches are padded to their largest point cloud,
increase GPU memory use, and change the number of optimizer updates. Use the same batch
size for every row of a comparison. Test evaluation remains at batch size `1`.

Outputs are written under `outputs/stage1_ablation_3/`. Training checkpoints live at
`<training_dataset>/best.ckpt`; individual test results live at
`<training_dataset>/by_test_dataset/<evaluation_dataset>/test_metrics.json`.
`matrix.json`, `matrix.csv`, and `matrix.md` contain all 16 cells, with rows denoting the
training dataset and columns denoting the test dataset. Five `matrix_*.png` heatmaps use
the same Stage 1 task metrics as Ablation 2. Cross-source checkpoint loading remains
strict everywhere else; Ablation 3 explicitly enables the evaluator's controlled
same-schema, same-layout dataset-mismatch mode for off-diagonal cells.

To render the combined KPConvX Ablation 1 test set again:

```bash
make visualize-stage1-test
# Run each requested view and write one subdirectory per view:
make visualize-stage1-test STAGE1_PCL_TYPES="full top_down side"
```

Expected: six-panel prediction/target previews and test metrics in
`outputs/stage1_ablation_1/combined/kpconvx/test_visualizations` for the default
single `[full]` view. A single derived view keeps the existing suffixed directory
behavior. Multiple views write `full/`, `top_down/`, and `side/` subdirectories
plus root metrics and a summary manifest. Both commands run in Docker;
`STAGE1_VIS_OUTPUT` can override the output directory. The underlying script accepts
`--pcl-types full top_down side`.

The two probability panels share a bottom Turbo colourbar. Junction detections are
shown explicitly: cyan dots with black outlines are ground truth, while magenta dots
with white outlines are probability-thresholded, spatially clustered predictions.
Both junction marker types use three times the skeleton-node radius.
These display-only centroids do not affect Junction F1 or any other evaluation metric.

Each selected cloud alone drives its encoder inference, metrics, and all six panels.
Derived input retains the labels of its saved points and uses the full plant's
ground-truth skeleton; point metrics cover only retained input points. The checkpoint
is evaluated as-is. PNG headers, metrics, and manifests identify each view. Missing
or stale partial clouds produce an error; regenerate the corresponding artifacts.
Do not use these test results to tune training settings or thresholds.

### 6. Cache encoder features or predictions

Prerequisite: Stage 1 checkpoint and prediction file.

```bash
python scripts/cache_stage_predictions.py --stage encoder \
  --checkpoint outputs/stage1_ablation_1/combined/kpconvx/best.ckpt \
  --input outputs/stage1_ablation_1/combined/kpconvx/smoke_predictions.pt \
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
  model.encoder.checkpoint=outputs/stage1_ablation_1/combined/kpconvx/best.ckpt
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
  model.encoder.checkpoint=outputs/stage1_ablation_1/combined/kpconvx/best.ckpt \
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
  model.encoder.checkpoint=outputs/stage1_ablation_1/combined/kpconvx/best.ckpt \
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
  input.path=data/dataset/plant_000001/full.npz \
  data.pcl_types=[full,top_down,side] \
  model.pipeline_checkpoint=outputs/joint/best.ckpt \
  inference.num_diffusion_samples=4 output.dir=outputs/inference/plant_000001
test -f outputs/inference/plant_000001/inference_manifest.json
```

One selected view writes the existing output contract directly. Multiple views write
one subdirectory per view and a root `inference_manifest.json`. A side run consumes
only `side.npz` points while keeping the full target contract. Raw CSV/PLY input
requires `data.pcl_types=[full]`. No ground-truth node count is used during inference.

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
  configs outputs/stage1_ablation_1 outputs/stage1_ablation_2 outputs/stage1_ablation_3 outputs/encoder outputs/diffusion outputs/graph outputs/parametric \
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
