# TomatoWUR v3 data integration

TomatoWUR is an external mount. Training and preprocessing never prompt for a download. The supported public v3 layout is:

```text
TomatoWUR/
  point_clouds/<plant_id>.csv
  ann_versions/<annotation_version>/
    annotations/<plant_id>/<plant_id>_labels.csv
    annotations/<plant_id>/<plant_id>_skeleton.csv
    json/{train,val,test}.json
  images/
  camera_poses/
```

The raw layout is read in place and is never reorganized. Processed data uses a shared,
dataset-agnostic collection rooted at `data/dataset`:

```text
data/dataset/
  manifest.json
  plant_000001/
    sample.npz
    sample.graph.json
    sample.params.json
    context.npz              # only when support context exists
  plant_000002/
    ...
```

One point cloud is one processed plant instance. Its globally unique `instance_id` and
`plant_id` are both the numbered folder name. `source_instance_id`, `dataset`, and
`source_plant_id` retain the source scan, source dataset, and biological plant identities.
This lets a time-series dataset store multiple scan instances for one biological plant
without putting source-specific directories in `data/dataset` or causing output collisions.
Biological source-plant splitting is enforced: two instances of one source plant cannot
cross train, validation, and test splits. `plant_count` counts the globally unique processed
plant instances; `source_plant_count` separately counts their biological source plants.

Before allocating a new folder, preprocessing scans both the root manifest and all existing
`plant_<number>` directories. It continues at the highest number plus one regardless of
which source dataset was processed first. A manifest reservation is written before each
point cloud is converted, so an interrupted run reuses its assigned folder. Completed
source instances with unchanged inputs and configuration are skipped on reruns.

Point CSV fields are metric `x,y,z`, byte-range `blue,green,red`, and `nx,ny,nz`. The loader converts BGR to canonical RGB `[0,1]`. Annotation aliases are accepted for semantic and instance columns; official semantics are background 0, leaf 1, main stem 2, support pole 3, and side stem 4. Skeleton CSV fields are `x_skeleton,y_skeleton,z_skeleton,vid,parentid,edgetype`; optional measured traits are `gt_int_length`, `gt_int_diameter`, `gt_ph_angle`, and `gt_lf_angle`. Missing traits remain NaN/masked.

Raw semantic value 255 denotes an unlabeled point and is converted to the internal ignore index (-100). Such points remain available to geometry-based auxiliary targets but are excluded from semantic cross-entropy.

## Official ground-truth skeletons

Stage 0 uses the corrected `0-paper-2Dto3D_improved` skeleton CSVs supplied with TomatoWUR v3. These are the paper's manually curated ground-truth skeletons after leaf-point removal and skeleton correction. Preprocessing does not rerun Xu skeletonisation and does not infer, remove, reconnect, resample, or reduce any annotated edge.

Every official skeleton currently contains 114–251 valid nodes and therefore fits K=256. The raw node order, coordinates, parent indices, and edge types are copied into the valid slots after root translation; remaining slots are padding. If a future annotation contains more than K nodes, preprocessing stops and asks for a larger K rather than changing the GT.

The skeleton CSV is decoded with TomatoWUR's upstream convention: coordinate row number is the node index, while each row's `parentid,vid,edgetype` fields form an edge-list entry and need not describe that same coordinate row. The cache therefore assigns `parent_index[vid] = parentid` and maps `<` to same-axis continuation and `+` to side-axis attachment. Original node IDs, parent IDs, and edge types are retained in sample metadata.

`scripts/visualize_dataset.py` renders X-Z, Y-Z, and X-Y views. Official GT parent edges are red.

Split JSON entries must resolve point-cloud, label, and skeleton paths. Official key names `file_name`, `sem_seg_file_name`, and `skeleton_file_name` are supported, as are explicit aliases. Alternate files/views of one plant must stay in one plant-level split.

## Deterministic conversion

`scripts/prepare_tomatowur.py` performs these operations:

1. Validate required files, row counts, finite coordinates, and rooted-tree connectivity.
2. Translate the skeleton root to the origin and store the exact inverse 4×4 transform.
3. Remove semantic class 3 from plant learning and store support-pole points in a separate context NPZ.
4. Voxel-downsample deterministically and retain original-point indices.
5. Validate the official rooted tree and copy all GT nodes, parents, and edge types unchanged.
6. Pad to K, derive normalized parent flow, organ type, topology role, and support-distance visibility.
7. Fit stem/leaf parameter targets from graph chains and labelled scan support.
8. Reserve the next global plant number, then write one versioned JSON/NPZ cache directory per point-cloud instance and update the shared manifest.

Processed files contain the canonical tensor fields documented in
[MODEL_CONTRACTS.md](MODEL_CONTRACTS.md). Training selects a source with
`data.dataset=<source_id>` or uses `data.dataset=combined`. A checkpoint records the
complete selected source/preprocessing-hash signature and is rejected when that contract
or K differs.

Fruit proposals for an instance belong in its numbered folder as `fruit_pseudo.npz`; no auxiliary directory is created at the dataset root. The included script writes the external detector contract under `outputs/fruit_pseudo` but intentionally fabricates no fruit supervision. Predictions below `fruit.min_confidence` are excluded, and fruit never enters primary metrics.
