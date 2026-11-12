# Model contracts

Stage 1 uses dataclasses in `tomato_recon.data.schemas`. DiffTS Stage 2 has its own target-free input and variable-length output contract; disk formats remain versioned JSON/NPZ/PT structures.

AG-GFM adds a separate `AGGFMInput` adapter and `Prediction` output. Its continuous
state is `[slots,4]` (XYZ metres and an existence logit); its output exposes
`node_xyz`, `parent_flow`, `parent_index`, `edges`, source slots and a validated
`PlantGraph`. Checkpoints use `aggfm-stage2-v1` with architecture version 1 and
embed the complete Stage 1 encoder including all prediction heads. No legacy
DiffTS schema changes. See [AG-GFM contracts and configuration](AGGFM.md).

## `PlantSample`

- Point tensors: `xyz/rgb/normals [N,3]`, `semantic/instance/point_valid [N]`.
- Fixed skeleton tensors: `node_xyz/parent_flow [K,3]`, `node_valid/parent_index/organ_type/topology_role/visibility [K]`.
- Optional typed `PlantGraph` and `ParametricPlant`, plus inverse transform, hashes, traits, and provenance in metadata.

## `EncoderOutput`

The backbone returns per-input-point features, optional multiscale feature maps, and a global feature. `PointEncoder` adds semantic, skeleton, centreline-offset, and junction heads. Optional backbone packages are imported only after selection.

## DiffTS Stage 2 prediction

`DiffTSPrediction` contains `node_xyz [M,3]`, normalized `parent_flow [M,3]`,
`parent_index [M]`, `edges [M-1,2]`, and `predicted_count=M` where
`2 <= M <= 256`. The supervised count head determines M from conditioning
features at inference; no target count is read. Direction-weighted skeleton
postprocessing creates one rooted tree. Standalone inference writes these
fields to `skeleton.json` and writes nodes and edges to `skeleton.ply`.

The `points` conditioner accepts XYZ, RGB, and RGB availability. The
`points_stage1` conditioner uses aligned point and global features from the
all-view KPConvX backbone, followed by a trainable sparse adapter. The
backbone is frozen by default and can be fine tuned. Training uses all three processed
views and each plant's full skeleton; standalone inference accepts only a
processed `top_down.npz` and does not open target fields.

## `PlantGraph`

JSON schema 1.0 records Z-up/metres, one root, typed/role/visibility/confidence nodes, parent/child typed edges, organs, traits, post-processing corrections, and provenance. `validate()` enforces unique IDs, one parent per non-root, connectivity, and acyclicity. The graph is not restricted to binary branching or an L-system.

## Parametric and geometry contracts

`OrganParameters` contains attachment transform, spline controls, taper radii, leaf width/bend coefficients, optional fruit radii, confidence, visibility, and source nodes. `PlantGeometry` contains one or more validated triangle meshes plus optional skeleton edges and visibility weights. Geometry construction stays in PyTorch; USD authoring consumes detached values.

## Checkpoints

The `points` checkpoint retains schema `diffts-stage2-v1`. The new
`points_stage1` adapter checkpoint uses `diffts-stage2-v2` and embeds the
backbone weights, its source checkpoint hash, and the freeze setting. Both
store the variant, resolved config, maximum count, all training views, dataset
compatibility signature, source commit, and metrics. Loading rejects variant,
node-limit, dataset, and architecture mismatches. Existing `points_stage1`
v1 checkpoints cannot resume the adapter architecture. The variants use
separate checkpoint directories. Stages 3–5 await integration
with this variable-length contract.
