# OpenUSD export

`USDPlantExporter` authors metres (`metersPerUnit=1.0`) and Z-up only. Invalid unit/axis configurations fail immediately. The routine output is `plant.usd`; `plant_debug.usda` is the text/debug companion.

```text
/World/TomatoPlant
  /Geometry/MainStem/*
  /Geometry/SideStems/*
  /Geometry/Leaves/*
  /Geometry/Fruits/*          optional, confidence-supported
  /Skeleton                   BasisCurves
  /Graph                      graph JSON asset reference
  /Materials
  /Collision                  optional
  /Metadata
/World/Context/SupportPole    optional non-plant context
```

Meshes carry points, triangle counts/indices, normals when available, material bindings, organ ID/type, parent organ, confidence, visibility, source nodes, and parameter JSON reference. Asset metadata carries plant/dataset/cultivar, schema, preprocessing and checkpoint hashes, git commit, and coordinate transform.

Install `.[usd]` or use the `usd-export` container to create binary crate `.usd` and reopen it through `pxr`. When `pxr` is absent, the same API writes valid USDA text at the `.usd` path, writes the debug `.usda`, and records a visible fallback warning; this makes CPU-only smoke tests possible without pretending the result is binary.

Static validation checks stage openability (when `pxr` is present), units, axis, required plant root, and non-empty mesh data. The optional Isaac Sim 6.0.1 Compose profile performs the downstream check and writes JSON. Physics is disabled by default; enabling collisions creates the reserved hierarchy for deliberately simplified static collision assets rather than treating visual leaf meshes as colliders.

