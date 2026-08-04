# TomatoWUR v3 data integration

TomatoWUR is an external mount. Training and preprocessing never prompt for a download. The supported public v3 layout is:

```text
TomatoWUR_v3/
  point_clouds/<plant_id>.csv
  ann_versions/<annotation_version>/
    annotations/<plant_id>/<plant_id>_labels.csv
    annotations/<plant_id>/<plant_id>_skeleton.csv
    json/{train,val,test}.json
  images/
  camera_poses/
```

Point CSV fields are metric `x,y,z`, byte-range `blue,green,red`, and `nx,ny,nz`. The loader converts BGR to canonical RGB `[0,1]`. Annotation aliases are accepted for semantic and instance columns; official semantics are background 0, leaf 1, main stem 2, support pole 3, and side stem 4. Skeleton CSV fields are `x_skeleton,y_skeleton,z_skeleton,vid,parentid,edgetype`; optional measured traits are `gt_int_length`, `gt_int_diameter`, `gt_ph_angle`, and `gt_lf_angle`. Missing traits remain NaN/masked.

Raw semantic value 255 denotes an unlabeled point and is converted to the internal ignore index (-100). Such points remain available to geometry-based auxiliary targets but are excluded from semantic cross-entropy.

Split JSON entries must resolve point-cloud, label, and skeleton paths. Official key names `file_name`, `sem_seg_file_name`, and `skeleton_file_name` are supported, as are explicit aliases. Alternate files/views of one plant must stay in one plant-level split.

## Deterministic conversion

`scripts/prepare_tomatowur.py` performs these operations:

1. Validate required files, row counts, finite coordinates, and rooted-tree connectivity.
2. Translate the skeleton root to the origin and store the exact inverse 4×4 transform.
3. Remove semantic class 3 from plant learning and store support-pole points in a separate context NPZ.
4. Voxel-downsample deterministically and retain original-point indices.
5. Resample every skeleton edge at metric spacing while preserving original endpoints.
6. If M>K, retain root/junction/tip nodes, allocate remaining samples along maximal chains by arc length, and reconnect each retained node to its nearest retained ancestor.
7. Pad to K, derive normalized parent flow, organ type, topology role, and support-distance visibility.
8. Fit stem/leaf parameter targets from graph chains and labelled scan support.
9. Write plain versioned JSON/NPZ caches and a manifest with source/config/cache hashes, counts, truncation rate, label map, and warnings.

Processed files contain the canonical tensor fields documented in [MODEL_CONTRACTS.md](MODEL_CONTRACTS.md). A cache is rejected when checkpoint preprocessing hash or K differs.

Fruit proposals belong under `<processed_root>/fruit_pseudo/`. The included script writes the explicit pseudo-label cache contract but intentionally fabricates no fruit supervision. An external offline detector may populate `points` and `confidence`; predictions below `fruit.min_confidence` are excluded, and fruit never enters primary metrics.
