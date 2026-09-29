# Tomato plant reconstruction: Stage 2 DiffTS

This repository prepares tomato point clouds, trains a Stage 1 point encoder, and now trains two DiffTS Stage 2 skeleton models. The **points** variant conditions on XYZ and available RGB; **points_stage1** uses an all-view Stage 1 backbone as its feature encoder, followed by a trainable sparse adapter. Stage 1 is frozen by default and can be fine tuned during Stage 2. Both infer skeleton nodes and parent directions from saved top-down point clouds. Stage 3 graph, Stage 4 parametric geometry, and Stage 5 joint inference commands currently report that DiffTS integration is pending.

Stage 2 adapts the [DiffTS implementation](https://github.com/PRBonn/DiffTS/tree/a1b75f5c4804ac018c553dcc6cf9a78bd518d8e4) to the existing CUDA 12.1/spconv environment. The sparse U-Net, weighted scan-neighbour conditioning, six-channel noise prediction, classifier-free guidance, and DPM solver follow DiffTS; a supervised count head removes its evaluation-time dependence on target node counts. Ported code retains the upstream MIT attribution.

## What is implemented

- Processed TomatoWUR, TomatoPGT, and Pheno4D scans with full, top-down, and side views. Existing train/validation/test splits are preserved.
- KPConvX, PointNeXt, and Sonata Stage 1 encoder training and checkpoint workflows.
- Two separate DiffTS Stage 2 training and target-free top-down inference workflows, with saved XYZ/flow/tree outputs and source/view evaluation metrics.
- Deferred Stage 3–5 entry points with an explicit integration error.

## Smoke test

The Stage 2 smoke run requires the prepared combined dataset, the all-view
KPConvX Stage 1 checkpoint, and an NVIDIA GPU. It exercises both DiffTS
variants at the production 8,192-point input size with short sampling. Run it
in the training image:

```bash
docker compose -f docker/docker-compose.yml run --rm train bash scripts/smoke_test_all.sh
```

For CPU-only unit tests in a local Python environment, install `.[dev]` and
run `python -m pytest -q`.

## Required ordered training and execution procedure

Run the following steps in order. Stage 1 and each DiffTS variant own separate checkpoint directories and accept `--resume PATH`.

### 1. Build the training image and verify GPU access

Prerequisite: Docker, BuildKit, NVIDIA Container Toolkit, and a compatible NVIDIA driver.

```bash
docker compose -f docker/docker-compose.yml build train
docker compose -f docker/docker-compose.yml run --rm train python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: the cached multi-stage image builds and the check prints `True` for CUDA on a GPU host. Verify the resolved image digest with `docker image inspect tomato-parametric-reconstruction-train --format '{{json .RepoDigests}}'` and archive it with the final run.

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

Stage 1 has three complementary ablations. Ablation 1 compares all three encoder
architectures on one source-dataset selection and one ordered point-cloud configuration.
The default uses all sources and the full point cloud:

```bash
make stage1-ablation-1
make stage1-ablation-1 STAGE1_ABLATION1_RESUME=--resume
make stage1-ablation-1 STAGE1_ABLATION_BATCH_SIZE=2
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
make stage1-ablation-2 STAGE1_ABLATION_BATCH_SIZE=2
```

Outputs are written under `outputs/stage1_ablation_2/`. `matrix.json`, `matrix.csv`, and
`matrix.md` contain the complete results. Five `matrix_*.png` heatmaps cover semantic
mIoU, skeleton F1, centreline-offset score, junction F1, and weighted overall score.
Rows are training configurations and columns are test configurations. Pair and triple
configurations pool their selected views as separate samples. All Stage 1 ablations run
in the Docker training service and require a visible CUDA GPU.

Ablation 3 measures cross-dataset generalization with the default KPConvX encoder. It
trains four checkpoints using `full`, `top_down`, and `side` as separate samples: one
checkpoint for each source dataset and one for their combined training split. Every
checkpoint is then evaluated on the held-out test split for all four dataset selections,
producing a 4-by-4 matrix. Validation from the training dataset selects each best epoch;
test results are never used for selection:

```bash
make stage1-ablation-3
make stage1-ablation-3 STAGE1_ABLATION3_RESUME=--resume
make stage1-ablation-3 STAGE1_ABLATION_BATCH_SIZE=2
```

All three Stage 1 ablations use the shared `STAGE1_ABLATION_BATCH_SIZE` setting, which
defaults to `1`. Larger batches are supported, but mixed view batches are padded to their
largest point cloud, increase GPU memory use, and change the number of optimizer updates.
Use the same value when invoking each ablation. Test evaluation remains at batch size `1`.

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

### 7. Train Stage 2: DiffTS

The default processed dataset has 92 training, 29 validation, and 42 test plants. Each training plant contributes full, top-down, and side inputs, for 276 training examples. Targets are the same existing full skeleton for each plant in every view. Run `python scripts/audit_diffts_data.py` to verify source alignment, split/view counts, and colour availability. TomatoWUR and TomatoPGT retain RGB; Pheno4D uses an explicit unavailable-colour flag. Checkpoint selection uses top-down validation skeleton Chamfer distance. The two runs use identical data, sampling, seeds, and DiffTS hyperparameters.

```bash
# Run a short CUDA check for both variants first.
docker compose -f docker/docker-compose.yml run --rm train bash scripts/smoke_test_all.sh

# Then train each variant to validation convergence.
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_diffusion model.diffusion.variant=points

docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_diffusion model.diffusion.variant=points_stage1 \
  output.dir=outputs/diffts/points_stage1_adapter
```

Training writes `outputs/diffts/points/{best,last}.ckpt` and `outputs/diffts/points_stage1_adapter/{best,last}.ckpt`. The new Stage 1 adapter starts from the all-view checkpoint configured by `model.encoder.checkpoint`; its own checkpoint embeds the backbone weights. The default uses 8,192 sampled conditioning points, batch size one, mixed precision, four-step gradient accumulation, a 200-epoch ceiling, validation every five epochs, and the currently configured patience of 100 validation checks. Set `trainer.freeze_stage1=false` to fine tune the backbone at `trainer.stage1_learning_rate` (default `0.00001`). Resume with `--resume outputs/diffts/points_stage1_adapter/last.ckpt`; the freeze setting may change on resume. Existing `points_stage1` checkpoints use the previous architecture and cannot resume this run.

### 8. Evaluate the held-out top-down views

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.evaluate_diffts \
  --checkpoint outputs/diffts/points/best.ckpt \
  --output outputs/diffts/points/test_top_down.json

docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.evaluate_diffts \
  --checkpoint outputs/diffts/points_stage1_adapter/best.ckpt \
  --output outputs/diffts/points_stage1_adapter/test_top_down.json
```

Reports contain node and sampled-skeleton Chamfer distances, node-count error, and per-source summaries for all 42 held-out plants. Use `--split val --all-views` to inspect validation metrics for all three views.

### 9. Infer one saved top-down scan

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.infer_diffts \
  --checkpoint outputs/diffts/points/best.ckpt \
  --input data/dataset/plant_000001/top_down.npz \
  --output outputs/diffts/inference/plant_000001
```

Inference reads only the saved `top_down.npz` input and its metadata. It writes `skeleton.json` with predicted node count, XYZ, parent directions, and parent indices, plus `skeleton.ply` with tree edges. It never reads graph or node targets. A legacy Stage 2 checkpoint cannot load as a DiffTS checkpoint.

## Configuration and checkpoints

Stage 2 settings live in `configs/diffusion/default.yaml`. `points` retains the `diffts-stage2-v1` checkpoint schema; new `points_stage1` adapter checkpoints use `diffts-stage2-v2` and embed the Stage 1 backbone weights, its source checkpoint hash, and the freeze setting. Both store the resolved config, training views, and dataset compatibility signature. Loading rejects incompatible variants, data, and architecture versions.

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

The suite covers dataset contracts and hashes, Stage 1, DiffTS noising/sampling/count prediction/checkpoint compatibility, target-free top-down input loading, and existing geometry/export modules. GPU smoke runs exercise both DiffTS conditioning variants.
