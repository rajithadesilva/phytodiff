# Model contracts

All stage boundaries use dataclasses in `tomato_recon.data.schemas`; disk formats remain plain, versioned JSON/NPZ/PT checkpoint structures.

## `PlantSample`

- Point tensors: `xyz/rgb/normals [N,3]`, `semantic/instance/point_valid [N]`.
- Fixed skeleton tensors: `node_xyz/parent_flow [K,3]`, `node_valid/parent_index/organ_type/topology_role/visibility [K]`.
- Optional typed `PlantGraph` and `ParametricPlant`, plus inverse transform, hashes, traits, and provenance in metadata.

## `EncoderOutput`

The backbone returns per-input-point features, optional multiscale feature maps, and a global feature. `PointEncoder` adds semantic, skeleton, centreline-offset, and junction heads. Optional backbone packages are imported only after selection.

## `SkeletonPrediction`

Diffusion returns fixed `[B,K,3]` position and normalized parent flow, `[B,K]` existence logits/confidence, and an inference-only NMS mask. Ground-truth count is not required. `sample_id` records stochastic samples.

## `PlantGraph`

JSON schema 1.0 records Z-up/metres, one root, typed/role/visibility/confidence nodes, parent/child typed edges, organs, traits, post-processing corrections, and provenance. `validate()` enforces unique IDs, one parent per non-root, connectivity, and acyclicity. The graph is not restricted to binary branching or an L-system.

## Parametric and geometry contracts

`OrganParameters` contains attachment transform, spline controls, taper radii, leaf width/bend coefficients, optional fruit radii, confidence, visibility, and source nodes. `PlantGeometry` contains one or more validated triangle meshes plus optional skeleton edges and visibility weights. Geometry construction stays in PyTorch; USD authoring consumes detached values.

## Checkpoints

Each `best.ckpt` contains schema/stage/epoch, model and optimizer state, fully resolved configuration, label map, preprocessing hash, K, git commit/dirty flag, upstream checkpoint hashes, and metrics. Loading fails before state mutation when cache hash or K is incompatible.

