# Anatomically Guided Graph Flow Matching (AG-GFM)

AG-GFM is a separate Stage 2 backend. `STAGE2_MODEL` selects `diffts` (the default)
or `aggfm`; it does not select a Stage 1 backbone, a DiffTS conditioning variant,
a source dataset, or a point-cloud view. All commands below use the repository's
Docker training service. No new Python dependencies are required.

`make help` lists the workflow commands. Use `stage2-train`, `stage2-evaluate`,
`stage2-infer`, and `stage2-visualize` with `STAGE2_MODEL=aggfm`. Earlier command
names remain available as aliases with the same options and output paths.

## Architecture and supervised objective

1. **Anatomical seeds.** The complete trained Stage 1 `PointEncoder`, including
   semantic, skeleton, centreline-offset and junction heads, runs on the actual
   input points. No cached validation predictions are loaded. Observed seeds
   shift points by centreline offsets and use confidence-weighted farthest
   sampling, with a minimum spacing to avoid duplicate candidates. Latent learned
   queries combine the global feature with learned position/existence heads and
   Gaussian perturbations. Defaults allocate at most 192 observed and 64 latent
   candidates, for 256 slots. Short/duplicate input clouds can produce fewer
   observed candidates. Learned organ probabilities can embed anatomical classes.
2. **Structure-aware partial transport.** A SciPy Hungarian solver matches slots
   to unpadded full GT nodes. Costs combine Euclidean distance divided by
   `xyz_scale_m`, negative log organ probability, and L1 structural distance.
   Source descriptors predict `log(1 + root-path-length / xyz_scale_m)` and
   junction probability from seed conditioning features; target descriptors use
   the annotated parent hierarchy and role. These source predictions receive
   explicit supervised auxiliary losses. They are also defined without GT at
   inference. A square matrix with source and target dummy assignments allows
   both surplus slots and uncovered target nodes; each unmatched source or
   target costs `matching.unmatched_cost`. A match can be rejected when its cost
   exceeds the two unmatched penalties. Metrics record every cost component,
   matched counts, unmatched counts, and target coverage.
3. **Conditional graph flow.** Four query Transformer blocks default to width
   192, eight attention heads and dropout 0.1. They combine node self-attention,
   cross-attention to spatially pooled Stage 1 point features, local kNN geometric
   messages, sinusoidal time features, and soft time-conditioned parent context.
   The parent head scores each other slot. Self-connections are excluded;
   existence probabilities gate intermediate parent messages. The output heads
   predict velocity, parent logits, topology role and organ class.

State is `Z = [x, y, z, existence_logit]`. Training samples a uniform continuous
time, interpolates `Z_t = (1-t) Z_0 + t Z_1`, and regresses `Z_1-Z_0`. Matched
endpoints use GT XYZ and logit +3; unmatched endpoints retain the initial,
detached XYZ and use logit -3. Thus inactive coordinate targets remain defined.
XYZ velocity is divided by 0.1 metres and existence velocity by 6 before MSE.

The primary objective is
`w_flow L_flow + w_exist L_exist + w_parent L_parent + w_role L_role + w_geom L_geom`.
Existence BCE uses the one-step endpoint estimate `Z_t+(1-t)V`. Parent CE uses
the matched GT parent only when both child and parent were matched; root,
unmatched, self, and inactive connections are masked. Role CE ignores missing
labels. Endpoint node Chamfer uses matched predictions and the full valid target.
`L_geom` also includes configurable seed structural/organ auxiliaries, so the
descriptors used by the discrete matcher are learned despite non-differentiable
assignments. Unknown organ labels are excluded from semantic costs and CE.
Padded nodes and unmatched slots do not enter parent, role or geometry losses.
Loss weights are `1.0 / 0.2 / 0.2 / 0.1 / 0.5` by default; auxiliary weights live
under `model.aggfm.loss.structure` and `.organ`.

At inference, Euler or Heun integrates from 0 to 1 (20 Heun steps by default).
Existence selects nodes, with a configurable minimum count of one if none cross
the threshold. Final role probabilities choose the root. NetworkX's directed
maximum spanning arborescence implements Chu–Liu/Edmonds with incoming root edges
removed. This gives exactly one parent per non-root node, connectivity and
acyclicity. An optional edge-distance mask is relaxed explicitly, with a record
in `PlantGraph.postprocessing`, if it would prevent a connected tree.

## Modules and interfaces

| Module | Responsibility |
| --- | --- |
| `models/stage2.py`, `stage2.py` | Lazy backend factory and selected CLI dispatcher |
| `data/aggfm.py` | View adapter, annotation exposure, point-only input and split signatures |
| `models/flow_matching/seeds.py` | Anatomical observed/latent candidates |
| `models/flow_matching/matching.py` | Partial Hungarian OT and GT structural descriptors |
| `models/flow_matching/transformer.py` | Query attention, local graph messages, parent/role heads |
| `models/flow_matching/model.py` | Swappable seeds, pooled memory and vector field |
| `models/flow_matching/flow.py` | Interpolation, supervised endpoints, Euler/Heun ODE sampling |
| `models/flow_matching/topology.py` | Directed tree decoding and `PlantGraph` export |
| `models/flow_matching/losses.py` | Masked objectives and parent supervision |
| `models/flow_matching/runtime.py` | Complete Stage 1 extraction, checkpoint contracts and prediction |
| `train/train_aggfm.py` | Selected-data training and validation-only selection |
| `evaluation/aggfm.py`, `evaluate_aggfm.py` | Matched metrics and evaluation CLI |
| `infer_aggfm.py`, `visualize_aggfm.py` | Target-free prediction and held-out flow panels |

`create_stage2_model(name, cfg)` creates either backend lazily. The selected CLI
is `python -m tomato_recon.stage2 {train,evaluate,infer,visualize} --model aggfm`.
Existing `train_diffusion`, `train_diffts`, `evaluate_diffts`, `infer_diffts`, and
`scripts/visualize_stage2_predictions.py` keep their original interfaces.

## Dataset and view selection

The data adapter reuses `PointCloudViewDataset`. Every selected partial view uses
the same original full target. `organ_type`, `topology_role` and `visibility` are
read from valid node rows, with parent indices remapped accordingly. Original
cache annotations and normalization transforms are not rewritten.

Dataset choices are `tomatowur`, `tomatopgt`, `pheno4d`, `combined`. Views can be
any nonempty ordered subset of `full`, `top_down`, `side`. Duplicate/unknown views
are rejected. Actual manifest splits are used with no fixed counts; repeated
scans of a source plant crossing splits are rejected. If top-down is selected,
its validation sampled-skeleton Chamfer chooses `best.ckpt`. Otherwise
`trainer.validation_view` selects a training view (default: the first selected
view). Test results never choose checkpoints.

Pheno4D's RGB-unavailable flag is preserved, and unavailable RGB is zeroed before
Stage 1. Checkpoint loading verifies Stage 1 architecture metadata and strictly
loads every prediction head. Each selected source's preprocessing hashes must be
covered by the Stage 1 checkpoint; a combined Stage 1 checkpoint can condition
individual-source AG-GFM runs. Stage 1 padding capacity and AG-GFM slot capacity
are independent. The default encoder is pretrained KPConvX; the existing encoder
registry remains available for controlled experiments and CPU fixtures.

## Docker training commands

Prepare the data and Stage 1 checkpoint using the existing workflows first.
The default encoder checkpoint is the output of `make stage1-train`. To use an
existing ablation checkpoint, add
`STAGE2_ENCODER_CHECKPOINT=outputs/stage1_ablation_1/combined/kpconvx/best.ckpt`.

```bash
make stage2-train STAGE2_MODEL=aggfm

make stage2-train STAGE2_MODEL=aggfm \
  STAGE2_DATASET=tomatowur STAGE2_PCL_TYPES="top_down"

make stage2-train STAGE2_MODEL=aggfm \
  STAGE2_DATASET=tomatopgt STAGE2_PCL_TYPES="full side"

make stage2-train STAGE2_MODEL=aggfm \
  STAGE2_DATASET=combined STAGE2_PCL_TYPES="full top_down side"

make stage2-train STAGE2_MODEL=aggfm \
  STAGE2_DATASET=pheno4d STAGE2_PCL_TYPES="side"

# The explicit target selects AG-GFM without another model variable.
make stage2-train-aggfm STAGE2_DATASET=tomatowur STAGE2_PCL_TYPES="full"
```

`STAGE2_EPOCHS=200`, `STAGE2_MAX_POINTS=8192`, `STAGE2_MAX_NODES=256`,
`STAGE2_BATCH_SIZE=1`, `STAGE2_LEARNING_RATE=0.0001`,
`STAGE2_FLOW_STEPS=20`, `STAGE2_FREEZE_STAGE1=true`, and `STAGE2_STAGE1_LR=0.00001`
are configurable. Batches accumulate gradients across variable-size plants,
then perform one optimizer update; no padded candidate attention is required.
The trainer uses float32 with gradient clipping and a validation-driven LR
scheduler. `STAGE2_PROCESSED_ROOT` overrides the processed data directory.

Validation runs the flow model on the selected device. Pairwise distances and
neighbor indices are shared across transformer blocks within each forward pass.
Intermediate ODE steps compute velocity without the final parent, role and organ
heads; the final pass still computes every head for reconstruction. Exact tree
decoding and Hungarian assignment remain on the CPU, with arrays transferred in
bulk to avoid scalar CUDA synchronization inside Python loops. These optimizations
preserve the solver, step count and checkpoint parameter layout.

All architecture, matching, loss and ODE settings live in
`configs/flow_matching/aggfm.yaml`, independent of DiffTS. Use
`STAGE2_AGGFM_CONFIG=path/to/custom.yaml`, or append dotlist settings:

```bash
make stage2-train-aggfm STAGE2_DATASET=tomatowur STAGE2_PCL_TYPES="side" \
  STAGE2_MAX_NODES=128 \
  STAGE2_AGGFM_OVERRIDES="model.aggfm.observed_seeds=96 model.aggfm.latent_seeds=32 model.aggfm.blocks=6 model.aggfm.matching.structure=0.5"

make stage2-train-aggfm STAGE2_FREEZE_STAGE1=false \
  STAGE2_AGGFM_OUTPUT=outputs/aggfm/combined/full_top_down_side_finetuned
```

Observed plus latent counts must be positive and at most `STAGE2_MAX_NODES`.
Change both counts when reducing that capacity below the default split.
For a direct Docker entry point:

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -u -m tomato_recon.train.train_aggfm \
  data.dataset=tomatowur 'data.pcl_types=[top_down]' \
  model.encoder.checkpoint=outputs/encoder/combined/kpconvx/best.ckpt
```

## Checkpoints and resume

Default checkpoint directories are `outputs/aggfm/<dataset>/<ordered_views>/`.
`STAGE2_AGGFM_OUTPUT` overrides the training directory. An existing directory with
checkpoints requires explicit resume, preventing accidental overwrites.

```bash
make stage2-train STAGE2_MODEL=aggfm \
  STAGE2_DATASET=tomatowur STAGE2_PCL_TYPES="top_down" \
  STAGE2_RESUME=outputs/aggfm/tomatowur/top_down/last.ckpt
```

`STAGE2_RESUME` is a path, not a string containing `--resume`. Direct training
entry points accept `--resume PATH`. Resume restores epoch, best validation
score, early-stop counter, optimizer, LR scheduler and Python/NumPy/PyTorch/CUDA
RNG states. AG-GFM checkpoints use schema `aggfm-stage2-v1`, architecture v1, and
embed both AG-GFM and complete Stage 1 weights. They retain the resolved config,
Stage 1 source path/SHA-256, selected dataset/views, preprocessing compatibility,
exact plant/split signature and random seed. Scaler state is supported by the
checkpoint API when supplied; the current float32 trainer does not use one.

Resume rejects model, loss/matching configuration, dataset/view, preprocessing,
split membership, seed, freeze/batch/LR and validation sampling incompatibility.
Epoch limits and output paths may change. Resume and inference use embedded
Stage 1 weights, so the original Stage 1 file is not needed after the run begins.
DiffTS checkpoints retain their original schema and cannot load as AG-GFM.

## Evaluation, inference and visualisation

`STAGE2_CHECKPOINT` defaults to the selected output directory's `best.ckpt`.
Supply the same dataset/views used for training to resolve that path, or set an
explicit checkpoint. A combined checkpoint can evaluate any of its source
datasets. `STAGE2_SELECTED_VIEWS` optionally changes evaluation/visualisation
views without changing the training-directory lookup. `STAGE2_SPLIT` is `val`
or `test`. Output paths are individually configurable.

```bash
# Evaluate an individual-source model.
make stage2-evaluate STAGE2_MODEL=aggfm \
  STAGE2_DATASET=tomatowur STAGE2_PCL_TYPES="top_down" STAGE2_SPLIT=test

# Evaluate a combined model on a selected source/view.
make stage2-evaluate STAGE2_MODEL=aggfm STAGE2_DATASET=pheno4d \
  STAGE2_CHECKPOINT=outputs/aggfm/combined/full_top_down_side/best.ckpt \
  STAGE2_SELECTED_VIEWS="side" STAGE2_SPLIT=test \
  STAGE2_EVAL_OUTPUT=outputs/aggfm/combined/full_top_down_side/pheno4d_side_test.json

# Target-free input; set the real processed plant path.
make stage2-infer STAGE2_MODEL=aggfm \
  STAGE2_DATASET=tomatowur STAGE2_PCL_TYPES="top_down" \
  STAGE2_INPUT=data/dataset/plant_000001/top_down.npz STAGE2_INFER_VIEW=top_down \
  STAGE2_INFER_OUTPUT=outputs/aggfm/tomatowur/top_down/inference/plant_000001

make stage2-visualize STAGE2_MODEL=aggfm \
  STAGE2_DATASET=tomatowur STAGE2_PCL_TYPES="top_down" STAGE2_SPLIT=val \
  STAGE2_SELECTED_VIS_COUNT=3 STAGE2_SELECTED_VIS_FRAMES=5 STAGE2_FLOW_STEPS=20
```

Equivalent direct inference in Docker is:

```bash
docker compose -f docker/docker-compose.yml run --rm train \
  python -m tomato_recon.infer_aggfm \
  --checkpoint outputs/aggfm/tomatowur/top_down/best.ckpt \
  --input data/dataset/plant_000001/top_down.npz --view top_down \
  --steps 20 --seed 42 --output outputs/aggfm/tomatowur/top_down/inference/plant_000001
```

Standalone input is an existing-contract NPZ containing `plant_id`, `xyz`, `rgb`,
`normals`, `point_valid`, and `metadata_json`. No node, parent, label or graph
arrays/files are opened. Partial views require the current view-algorithm
metadata. Points must already be root-centred in metres with Z up using the
existing preprocessing contract; arbitrary raw PLY input is not handled here.
The inverse normalization transform is retained in the graph's source metadata.
Exported XYZ remains in the same normalized metre frame as DiffTS.

Evaluation reports node Chamfer, sampled-skeleton Chamfer (eight samples per edge
plus endpoints), absolute node-count error, junction precision/recall/F1,
parent accuracy, edge precision/recall, root accuracy/distance, tree validity,
and synchronized inference runtime including Stage 1 conditioning. Hungarian
Euclidean matching with a 2 cm acceptance threshold aligns node IDs before
topology metrics; unmatched GT edges count as misses. Results include per-source
and per-view aggregates. Original visibility annotations support separately
named annotated-group target recall/distances, without claiming they describe
visibility under a newly selected partial view.

Pillow panels display the input cloud, observed/latent seeds, selected
intermediate flow states coloured by existence, final active nodes/tree, and GT.
GT is loaded only by evaluation/visualisation. The full trajectory is also saved
as NPZ for later analysis.

```text
outputs/aggfm/<dataset>/<view_combination>/
  resolved_config.yaml
  best.ckpt
  last.ckpt
  metrics.json                  # losses and individual matching costs
  validation.json
  evaluation_test.json          # STAGE2_EVAL_OUTPUT
  inference/<plant>/            # choose STAGE2_INFER_OUTPUT per plant
    skeleton.json
    skeleton.ply
    graph.json                  # existing PlantGraph schema
  visualizations/<split>/<view>/<plant>/
    flow.png
    trajectory.npz
    graph.json
```

## DiffTS compatibility and limitations

- `make stage2-train` still runs both DiffTS variants. `stage2-train-diffts-points`,
  `stage2-train-diffts-stage1`, `stage2-visualize-diffts-test`, legacy checkpoint paths,
  and DiffTS internals are unchanged. Selected commands default to DiffTS unless
  `STAGE2_MODEL=aggfm` is given. Legacy DiffTS training retains combined/all-view
  restrictions and historical split-count checks; its selected evaluation
  supports top-down or all three ordered views, standalone inference requires
  top-down, and selected visualisation uses the test split and runs the original
  visualisation once per selected view.
  Unsupported selections produce errors instead of changing legacy behavior.
- This is continuous existence transport with conditional parent prediction,
  not a discrete graph-flow solver. `flow.py` and `topology.py` can be replaced
  independently for later discrete solvers.
- Transport uses hard partial Hungarian assignments, not entropic OT. Source
  structural descriptors approximate hierarchy before a graph exists. The
  endpoint estimate used for auxiliary losses is one vector-field evaluation,
  while evaluation integrates the full ODE. These are numerical/modeling
  approximations rather than GT-assisted inference.
- No additional GT nodes are synthesized beyond the original cache capacity.
  Changing AG-GFM capacity does not alter the original annotations. Uncovered
  targets are diagnosed explicitly when candidates are insufficient or rejected.
- Visibility in exported nodes describes seed provenance, with this basis stored
  in graph metadata. Cached visibility labels originate from the full scan and
  cannot establish per-partial-view observed versus occluded branch accuracy.
- Outputs are Stage 2 skeletons/graphs; downstream Stage 3–5 integration remains
  subject to the existing repository's pending integration. No full training
  experiment or reconstruction-quality benchmark is implied by smoke tests.
- Evaluation uses a local seed generator and dropout-free model evaluation.
  CUDA training reproducibility remains subject to PyTorch attention/kernel
  determinism limitations; the common deterministic setting uses `warn_only`.

## Tests

The synthetic nine-node branching fixture covers complete Stage 1 head loading,
seed differentiation, dummy assignments, structural-cost effects, flow/existence
targets, parent masking, directed optimal trees, all dataset/view combinations,
split isolation, checkpoint/RNG resume, target-free inference, geometry/topology
metrics and legacy Makefile behavior. Integration tests run one CPU training
batch, resume, evaluate, infer and render with temporary fixture data. The CUDA
test performs a KPConvX forward pass, backward pass, optimizer update and final
AG-GFM inference.

```bash
docker compose -f docker/docker-compose.yml run --rm \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 train \
  python -m pytest tests/unit/test_aggfm.py tests/integration/test_aggfm.py -q

docker compose -f docker/docker-compose.yml run --rm train \
  python -m pytest tests/integration/test_aggfm.py -k cuda -q

docker compose -f docker/docker-compose.yml run --rm train python -m pytest -q
```

The Docker `dev` target declares pytest and Ruff. If an older local image lacks
the repository's declared dependencies, rebuild it with
`docker compose -f docker/docker-compose.yml build train` before using the full
suite or lint tools. No dataset download or full training is part of these tests.
