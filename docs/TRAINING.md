# Training and execution

The stages are deliberately separate. Do not skip ahead, mix preprocessing hashes, change K between stages, or tune any setting on the test split. Use `--resume PATH` to resume the same stage; checkpoint validation rejects incompatible caches.

Every training stage reports epoch and batch progress, running component losses, ETA, and elapsed epoch time. Interactive terminals update one line in place; redirected or captured logs emit periodic progress lines.

## 1. Build Docker and verify GPU

- Prerequisite: Docker with BuildKit and NVIDIA Container Toolkit.
- Command: `docker compose -f docker/docker-compose.yml build train`, then `docker compose -f docker/docker-compose.yml run --rm train python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
- Expected: a cached image and CUDA `True` on a GPU host.
- Verify: record `docker image inspect ... --format '{{json .RepoDigests}}'` for the final archive.

## 2. Mount or download TomatoWUR v3 outside the image

- Prerequisite: access to TomatoWUR v3.
- Command: place or mount it at `data/TomatoWUR`; run `test -d data/TomatoWUR/point_clouds && test -d data/TomatoWUR/ann_versions`.
- Expected: original point clouds, annotations/splits, images, and camera poses remain outside the image.
- Verify: update `data.raw_root` if the mount differs. No project command downloads the data.

## 3. Preprocess and inspect fixed-K statistics

- Prerequisite: Step 2 and plant-level splits.
- Command: `docker compose -f docker/docker-compose.yml run --rm preprocess python scripts/prepare_tomatowur.py --config configs/data/tomatowur_v3.yaml`.
- Expected: deterministic NPZ/graph/parameter caches and one `manifest.json` containing the official train, validation, and test partitions.
- Verify: `split_counts` is `train: 35`, `val: 4`, `test: 5`, `skeleton_annotation_version` is `0-paper-2Dto3D_improved`, and `skeleton_modified_count` is zero. Also inspect source hashes, warnings, and the label map. Every GT skeleton must fit K=256 without resampling or reduction.

## 4. Visualise three plants

- Prerequisite: three processed plants.
- Command: `python scripts/visualize_dataset.py data/dataset --count 3 --output outputs/dataset_preview`.
- Expected: three PNG previews, each with X-Z, Y-Z, and X-Y projections. Red edges are copied directly from the official corrected GT.
- Verify: visually inspect roots, labels, support-pole separation, tips, junctions, scale, and connectivity.

## 5. Train Stage 1 encoder

- Prerequisite: verified caches and `make prepare-stage1-models` when running Sonata.
- Combined command: `python -m tomato_recon.train.train_encoder --config-name encoder data.dataset=combined output.dir=outputs/encoder/combined`.
- Individual command: replace `combined` with `tomatowur`, `tomatopgt`, or `pheno4d` and use a distinct output directory.
- Supported encoders: `pointnext`, `sonata_ptv3`, and `kpconvx`.
- Expected: the selected output directory contains `best.ckpt`, `last.ckpt`, metrics, run metadata, and `smoke_predictions.pt`. Training uses only `train`, validates on `val`, and selects `best.ckpt` by validation overall score.
- Verify: inspect `val_semantic_miou`, validation skeleton precision/recall, centreline offset MAE, junction F1, and checkpoint stage/hash/K fields. The test split is not loaded during training or checkpoint selection.

Visualise predictions from the trained Stage 1 checkpoint:

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python scripts/visualize_encoder_predictions.py \
  --checkpoint outputs/encoder/combined/best.ckpt \
  --processed-root data/dataset \
  --split test --dataset combined --count 3 \
  --output outputs/encoder/combined/test_visualizations
```

Run the three-model architecture comparison, live global progress, held-out evaluation,
and all test-plant visualizations with `make stage1-ablation-1`. It uses one source
selection and one ordered point-cloud view array. Reports and the validation-selected
`winner.json` are written under `outputs/stage1_ablation_1/`.

Run `make stage1-ablation-2` after Ablation 1. It retrains the winning architecture on
all seven nonempty combinations of `full`, `top_down`, and `side`, then evaluates the
seven checkpoints on all seven test configurations. The 49-cell JSON/CSV/Markdown
reports and five task heatmaps are written under `outputs/stage1_ablation_2/`.

Each plant PNG contains six front-view panels: input RGB, ground-truth semantics,
predicted semantics, ground-truth structure, predicted skeleton probability with
offset-corrected centreline points, and junction probability with detections. Both
probability panels use the shared Turbo colourbar. In the junction panel, small cyan
dots with black outlines mark ground-truth junctions and magenta dots with white
outlines mark visualization-only clustered prediction centroids; both use the same
marker size, three times the ground-truth skeleton-node radius. The command defaults
to `test` and writes
`metrics.json` for the selected plants. Use `--plant-id PLANT_ID` (repeatable) to
select exact plants, `--count 0` for all 5 test plants, or `--device cpu` to disable
GPU inference. Run this only after model selection is complete.

Every source retains its plant-level split. The combined selector is the union of those
source splits. Do not move plants between partitions, use test previews to tune thresholds,
or repeatedly choose models using test results.

## 6. Cache encoder predictions

- Prerequisite: Stage 1 output.
- Command: `python scripts/cache_stage_predictions.py --stage encoder --checkpoint outputs/stage1_ablation_1/combined/kpconvx/best.ckpt --input outputs/stage1_ablation_1/combined/kpconvx/smoke_predictions.pt --output outputs/cache/encoder`.
- Expected: prediction file plus a manifest tied to its checkpoint SHA-256.
- Verify: recalculate or inspect both hashes in `outputs/cache/encoder/manifest.json`.

## 7. Train both DiffTS Stage 2 variants

Stage 2 uses the prepared combined dataset: 92 training plants × 3 views, 29
validation plants × 3 views, and 42 held-out test plants. Each training view
uses the same existing full skeleton target for its plant; only the input point
cloud changes. Verify all partial-view source hashes, split/view counts, RGB
availability, and target-free partial-view files with
`python scripts/audit_diffts_data.py`. The report is saved to
`outputs/diffts/data_audit.json`. Checkpoint selection uses the 29
validation top-down views and skeleton Chamfer distance. The points variant
uses XYZ, RGB, and an RGB availability flag. The points_stage1 variant adds
frozen per-point and global features from the all-view KPConvX Stage 1
checkpoint. Stage 1 weights do not update during Stage 2 training.

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_diffusion model.diffusion.variant=points

docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_diffusion model.diffusion.variant=points_stage1
```

Outputs are `outputs/diffts/points/{best,last}.ckpt` and
`outputs/diffts/points_stage1/{best,last}.ckpt`, with resolved configs and
validation metrics in each directory. The default is 8,192 scan points,
batch size one, mixed precision, accumulation of four, a half-width sparse
network, 200 epochs maximum, validation every five epochs, and patience of
ten validation checks. Use `model.diffusion.max_points=4096` for both runs
if the 8 GB GPU preflight cannot accommodate 8,192 points.

A short preflight for either variant is:

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_diffusion \
  model.diffusion.variant=points trainer.fast_dev_run=true \
  model.diffusion.sample_steps=4 output.dir=outputs/diffts/preflight_points
```

Replace `points` with `points_stage1` and use a separate output directory for
the second preflight. Inspect finite training losses, a 2–256 node prediction,
and readable JSON/PLY inference output before production training.

## 8. Evaluate both frozen checkpoints

Evaluate each best checkpoint once on the 42 held-out top-down views after
validation-based model selection. Reports include node Chamfer, skeleton
Chamfer, and count absolute error overall and by source/view.

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.evaluate_diffts \
  --checkpoint outputs/diffts/points/best.ckpt \
  --output outputs/diffts/points/test_top_down.json

docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.evaluate_diffts \
  --checkpoint outputs/diffts/points_stage1/best.ckpt \
  --output outputs/diffts/points_stage1/test_top_down.json
```

## 9. Infer a saved top-down view

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.infer_diffts \
  --checkpoint outputs/diffts/points/best.ckpt \
  --input data/dataset/plant_000001/top_down.npz \
  --output outputs/diffts/inference/plant_000001
```

The standalone loader accepts only processed `top_down.npz` inputs and reads
point/view/colour metadata. It does not open the skeleton targets. It writes
`skeleton.json` with predicted count, variable-length XYZ, parent directions,
parent indices, and edges, plus a skeleton PLY. Checkpoint loading rejects
old Stage 2 files, variant mismatches, preprocessing mismatches, and Stage 1
checkpoint hash mismatches.

## Later stages

Stages 3–5 and the old joint inference command report that DiffTS integration
is pending. Their former fixed-slot checkpoint paths are retired.
