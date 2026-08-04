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
- Command: place or mount it at `data/raw/TomatoWUR_v3`; run `test -d data/raw/TomatoWUR_v3/point_clouds && test -d data/raw/TomatoWUR_v3/ann_versions`.
- Expected: original point clouds, annotations/splits, images, and camera poses remain outside the image.
- Verify: update `data.raw_root` if the mount differs. No project command downloads the data.

## 3. Preprocess and inspect fixed-K statistics

- Prerequisite: Step 2 and plant-level splits.
- Command: `docker compose -f docker/docker-compose.yml run --rm preprocess python scripts/prepare_tomatowur.py --config configs/data/tomatowur_v3.yaml`.
- Expected: deterministic NPZ/graph/parameter caches and `manifest.json`.
- Verify: inspect `sample_count`, `fixed_k_reduction_rate`, source hashes, warnings, label map, `skeleton_quality_review_count`, and `skeleton_quality_review_plant_ids` in the manifest. Review each per-plant `.quality.json` before selecting exclusions.

## 4. Visualise three plants

- Prerequisite: three processed plants.
- Command: `python scripts/visualize_dataset.py data/processed/v3_10mm_K256 --count 3 --output outputs/dataset_preview`.
- Expected: three PNG previews, each with X-Z, Y-Z, and X-Y projections. Processed edges are red, unresolved suspicious edges are magenta, and replacement parents are cyan.
- Verify: visually inspect roots, labels, support-pole separation, tips, junctions, scale, connectivity, every magenta edge, and every cyan `parent_id→child_id` repair.

## 5. Train Stage 1 encoder

- Prerequisite: verified caches and the chosen backbone dependency.
- Command: `python -m tomato_recon.train.train_encoder --config-name encoder model.encoder.name=pointnext`.
- Expected: `outputs/encoder/best.ckpt`, metrics, run metadata, and `smoke_predictions.pt`.
- Verify: inspect semantic mIoU, skeleton precision/recall, centreline offset MAE, junction F1, and checkpoint stage/hash/K fields.

## 6. Cache encoder predictions

- Prerequisite: Stage 1 output.
- Command: `python scripts/cache_stage_predictions.py --stage encoder --checkpoint outputs/encoder/best.ckpt --input outputs/encoder/smoke_predictions.pt --output outputs/cache/encoder`.
- Expected: prediction file plus a manifest tied to its checkpoint SHA-256.
- Verify: recalculate or inspect both hashes in `outputs/cache/encoder/manifest.json`.

## 7. Train Stage 2 diffusion

- Prerequisite: compatible encoder checkpoint.
- Command: `python -m tomato_recon.train.train_diffusion --config-name diffusion model.encoder.checkpoint=outputs/encoder/best.ckpt`.
- Expected: `outputs/diffusion/best.ckpt`, component losses, and predicted fixed-K node sets.
- Verify: inspect denoising/existence/flow/duplicate/bounds losses, retained counts, confidence, and unit/zero parent-flow norms.

## 8. Cache diffusion predictions

- Prerequisite: Stage 2 output.
- Command: `python scripts/cache_stage_predictions.py --stage diffusion --checkpoint outputs/diffusion/best.ckpt --input outputs/diffusion/smoke_predictions.pt --output outputs/cache/diffusion`.
- Expected: immutable predictions tied to one diffusion checkpoint.
- Verify: inspect `outputs/cache/diffusion/manifest.json`; never combine incompatible K or preprocessing hashes.

## 9. Train Stage 3 graph

- Prerequisite: encoder and diffusion checkpoints/predictions.
- Command: `python -m tomato_recon.train.train_graph --config-name graph model.encoder.checkpoint=outputs/encoder/best.ckpt model.diffusion.checkpoint=outputs/diffusion/best.ckpt`.
- Expected: `outputs/graph/best.ckpt` and `smoke_graph.json`; the curriculum moves from clean ground-truth nodes through perturbed/predicted-node conditions.
- Verify: one root, one connected component, zero cycles, `edges = nodes - 1`, valid types, and a continuous main-stem path.

## 10. Train Stage 4 parametric decoder

- Prerequisite: graph checkpoint and automatically fitted cache targets.
- Command: `python -m tomato_recon.train.train_parametric --config-name parametric model.graph.checkpoint=outputs/graph/best.ckpt`.
- Expected: `outputs/parametric/best.ckpt`, fitted parameter JSON, and geometry PLY.
- Verify: positive radii, finite vertices, in-range triangle indices, valid attachment IDs, and non-empty stem/leaf meshes.

## 11. Joint fine-tune

- Prerequisite: compatible best checkpoints from Stages 1–4.
- Command: `python -m tomato_recon.train.train_joint --config-name joint model.encoder.checkpoint=outputs/encoder/best.ckpt model.diffusion.checkpoint=outputs/diffusion/best.ckpt model.graph.checkpoint=outputs/graph/best.ckpt model.parametric.checkpoint=outputs/parametric/best.ckpt`.
- Expected: `outputs/joint/best.ckpt`, upstream checkpoint hashes, `staged_validation.json`, component metrics, and `smoke_visibility_weights.pt`.
- Verify: inspect occlusion-aware geometry, normal, skeleton, parameter, and radius losses; confirm the discrete decoder has no gradient and the conservative learning rate is resolved.

## 12. Evaluate the frozen test set once

- Prerequisite: frozen experiment configuration and complete test predictions.
- Command: `python -m tomato_recon.evaluate --processed-root data/processed/v3_10mm_K256 --predictions outputs/inference --output outputs/evaluation/metrics.json`.
- Expected: per-sample/aggregate skeleton, topology, and trait results; geometry/fruit are in `secondary_aggregate`.
- Verify: `primary_metric_groups` contains skeleton/topology/traits, geometry contains normal consistency when normals exist, and no test result was used for tuning.

## 13. Infer one sample and export

- Prerequisite: joint checkpoint and processed NPZ or isolated CSV/ASCII PLY.
- Command: `python -m tomato_recon.infer --config-name infer input.path=data/processed/v3_10mm_K256/samples/PLANT_ID.npz model.pipeline_checkpoint=outputs/joint/best.ckpt inference.num_diffusion_samples=4 output.dir=outputs/inference/PLANT_ID`.
- Expected: every file in the inference output contract, including graph, parameters, mesh, traits, uncertainty, preview, USD, and report.
- Verify: open JSON/PLY outputs and check `export_report.json`; ground-truth node count is not read by sampling.

## 14. Validate USD and optionally Isaac Sim

- Prerequisite: Step 13.
- Command: `python -m tomato_recon.export.isaac_validate outputs/inference/PLANT_ID/plant.usd --report outputs/inference/PLANT_ID/static_validation.json`; optionally run `docker compose -f docker/docker-compose.yml --profile isaac-validate run --rm isaac-validate`.
- Expected: metres/Z-up, `/World/TomatoPlant`, valid meshes/materials/metadata, debug curves, and a machine-readable report.
- Verify: report `valid=true`; optional rendering absence is not a schema failure.

## 15. Archive the run

- Prerequisite: final metrics and validation.
- Command: `tar -czf outputs/tomatowur_experiment_archive.tar.gz configs outputs/encoder outputs/diffusion outputs/graph outputs/parametric outputs/joint outputs/evaluation outputs/inference`.
- Expected: configs, environment/git state, container digest record, manifests/splits, seeds, checkpoint dependencies, metrics, and per-sample predictions.
- Verify: inspect with `tar -tzf ...`; keep the archive, data, weights, and secrets outside git.

## Smoke and resume

Every stage accepts the dedicated tiny configuration:

```bash
python -m tomato_recon.train.train_encoder --config configs/smoke/all.yaml
python -m tomato_recon.train.train_encoder --config configs/smoke/all.yaml \
  --resume outputs/smoke/encoder/best.ckpt
```

`trainer.fast_dev_run=true` performs one forward/backward step. Normal configurations iterate `trainer.max_epochs` across the processed split with deterministic shuffling.
