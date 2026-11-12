# Training and execution

The stages are deliberately separate. Do not skip ahead, mix preprocessing hashes, change K between stages, or tune any setting on the test split. Use `--resume PATH` to resume the same stage; checkpoint validation rejects incompatible caches.

Run `make help` for the Docker workflow commands, grouped as `data-*`, `stage1-*`,
`stage2-*`, and `check-*`. Earlier Makefile command names remain aliases with the
same variables, output paths, and execution behavior.

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

- Prerequisite: verified caches and `make stage1-prepare-models` when running Sonata.
- Combined command: `python -m tomato_recon.train.train_encoder --config-name encoder data.dataset=combined output.dir=outputs/encoder/combined`.
- Individual command: replace `combined` with `tomatowur`, `tomatopgt`, or `pheno4d` and use a distinct output directory.
- Supported encoders: `pointnext`, `sonata_ptv3`, and `kpconvx`.
- Expected: the selected output directory contains `best.ckpt`, `last.ckpt`, metrics, run metadata, and `smoke_predictions.pt`. Training uses only `train`, validates on `val`, and selects `best.ckpt` by validation overall score.
- Verify: inspect `val_semantic_miou`, validation skeleton precision/recall, centreline offset MAE, junction F1, and checkpoint stage/hash/K fields. The test split is not loaded during training or checkpoint selection.
- Make target: `make stage1-train` trains KPConvX on the combined dataset with all three views and writes `outputs/encoder/combined/kpconvx/best.ckpt`. Set `STAGE1_EPOCHS` or `STAGE1_TRAIN_OUTPUT` to change the run; `STAGE1_CHECKPOINT` points to its best checkpoint for the Makefile Stage 2 feature variant.

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
and all test-plant visualizations with `make stage1-compare-backbones`. It uses one source
selection and one ordered point-cloud view array. Reports and the validation-selected
`winner.json` are written under `outputs/stage1_ablation_1/`.

Run `make stage1-compare-views` after Ablation 1. It retrains the winning architecture on
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

The instructions in this section retain the legacy DiffTS workflow. To use the
separate AG-GFM backend, see [AG-GFM](AGGFM.md). Its Docker Makefile commands support
individual datasets and arbitrary view combinations:

```bash
make stage2-train-aggfm STAGE2_DATASET=tomatopgt STAGE2_PCL_TYPES="full side"
make stage2-train STAGE2_MODEL=aggfm STAGE2_DATASET=combined STAGE2_PCL_TYPES="full top_down side"
make stage2-train STAGE2_MODEL=aggfm STAGE2_RESUME=outputs/aggfm/combined/full_top_down_side/last.ckpt
```

AG-GFM loads the complete Stage 1 checkpoint at `STAGE2_ENCODER_CHECKPOINT`
(defaults to `STAGE1_CHECKPOINT`). Stage 1 remains frozen unless
`STAGE2_FREEZE_STAGE1=false` is selected for a new run. Only validation skeleton
Chamfer chooses the best checkpoint; top-down takes precedence whenever selected.
No fixed plant counts are imposed by this trainer.

Stage 2 uses the prepared combined dataset: 92 training plants × 3 views, 29
validation plants × 3 views, and 42 held-out test plants. Each training view
uses the same existing full skeleton target for its plant; only the input point
cloud changes. Verify all partial-view source hashes, split/view counts, RGB
availability, and target-free partial-view files with
`python scripts/audit_diffts_data.py`. The report is saved to
`outputs/diffts/data_audit.json`. Checkpoint selection uses the 29
validation top-down views and skeleton Chamfer distance. The points variant
uses XYZ, RGB, and an RGB availability flag. The points_stage1 variant uses
the all-view KPConvX Stage 1 backbone as its only feature encoder; a trainable
adapter projects its point and global features onto the sparse grid consumed
by the denoising U-Net. Stage 1 is frozen by default.

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_diffusion model.diffusion.variant=points

docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.train.train_diffusion model.diffusion.variant=points_stage1 \
  output.dir=outputs/diffts/points_stage1_adapter
```

Use `make stage2-train` to run both variants in sequence, or run each target separately. The Stage 1 target uses `STAGE1_CHECKPOINT` (by default, the output of `make stage1-train`). Set `STAGE2_MAX_POINTS` and `STAGE2_EPOCHS` for both runs. Set `STAGE2_FREEZE_STAGE1=false` to fine tune the backbone, with `STAGE2_STAGE1_LR` controlling its learning rate. To change the freeze setting on resume, pass `STAGE2_STAGE1_RESUME="--resume outputs/diffts/points_stage1_adapter/last.ckpt"` to the Stage 1 target.

Outputs are `outputs/diffts/points/{best,last}.ckpt` and
`outputs/diffts/points_stage1_adapter/{best,last}.ckpt`, with resolved configs
and validation metrics in each directory. New adapter checkpoints embed the
Stage 1 backbone and do not need the source checkpoint for inference. Old
`points_stage1` checkpoints cannot resume the new architecture. The default
is 8,192 scan points, batch size one, mixed precision, accumulation of four,
a half-width sparse network, 200 epochs maximum, validation every five epochs,
and the currently configured patience of 100 validation checks. At the
200-epoch cap, this patience setting does not trigger early stopping. Use
`model.diffusion.max_points=4096` if the 8 GB GPU cannot hold 8,192 points.

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

## 8. Evaluate both checkpoints

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
  --checkpoint outputs/diffts/points_stage1_adapter/best.ckpt \
  --output outputs/diffts/points_stage1_adapter/test_top_down.json
```

Visualise the full denoising trajectory for held-out test plants, saving 12 evenly spaced frames:

```bash
make stage2-visualize-diffts-test STAGE2_VIS_FRAMES=12
```

The Make target defaults to the `points` checkpoint and `top_down` test view; set `STAGE2_VIS_VARIANT=points_stage1` to use the new adapter checkpoint, or `STAGE2_VIS_VIEW=full` to change the view. The sampler runs the checkpoint's configured number of inference steps (normally 200). It saves exactly 12 frames per plant: `frame_001.png` is the first denoising result, `frame_012.png` is the final result, and the intervening frames are evenly spaced in sampler-step order. The images are under the selected variant output directory at `test_diffusion_frames/<view>/frames_012_steps_200/<plant_id>/`; `manifest.json` records each frame's actual step and diffusion timestep. Use `STAGE2_VIS_SAMPLE_STEPS` only when you intentionally want a shorter sampling trajectory. The previous `STAGE2_VIS_STEPS` Make variable remains an alias for the number of saved frames.

Each frame shows large front X-Z prediction and ground-truth skeleton panels over the input scan. Both panels use one fixed scale from that plant's input scan and GT, so denoising outliers cannot shrink the visible plant; the footer counts predicted nodes outside the view. Set `STAGE2_VIS_COUNT=3` for a short preview. The script also accepts repeatable `--plant-id` arguments to select exact test plants.

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
