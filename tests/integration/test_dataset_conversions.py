from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from tomato_recon.data.pheno4d import (
    PHENO4D_COORDINATE_FRAME_VERSION,
    _normalised_to_source_mm,
    _stem_centerline,
    pheno4d_canonical_orientation,
    preprocess_pheno4d,
)
from tomato_recon.data.processed import ProcessedPlantDataset
from tomato_recon.data.schemas import OrganType, SemanticClass
from tomato_recon.data.tomatopgt import preprocess_tomatopgt
from tomato_recon.data.top_down import file_sha256


def _write_ply(path: Path, rows: list[tuple[float, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(rows)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar class\n"
        "end_header\n"
    ).encode("ascii")
    payload = b"".join(struct.pack("<ffffffBBBB", *row) for row in rows)
    path.write_bytes(header + payload)


def create_tomatopgt_fixture(root: Path) -> None:
    cultivar = root / "CULTIVAR_A"
    rows: list[tuple[float, ...]] = []
    annotations: list[str] = []
    source_classes = [0, 2, 7, 9, 1] * 8
    for index, source_class in enumerate(source_classes):
        z = index * 0.002
        x = 0.015 * (source_class in {1, 9})
        y = 0.001 * (index % 3)
        rows.append((x, y, z, 1.0, 0.0, 0.0, 20, 140, 30, 0))
        instance = 2 if source_class in {1, 9} else 0
        annotations.append(
            f"{x:.6f} {y:.6f} {z:.6f} 20 140 30 {source_class} {instance} 0 0 0"
        )
    point_path = cultivar / "CA_R_PC" / "CA_R_01012025.ply"
    _write_ply(point_path, rows)
    annotation_path = cultivar / "CA_ANNOTATED_TXT" / "CA_01012025_annotated_ext.txt"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text("\n".join(annotations) + "\n", encoding="utf-8")
    graph = {
        "meta": {
            "transform": {
                "translate": [0.0, 0.0, 0.0],
                "rotate": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "scale_div": 1.0,
            }
        },
        "nodes": [
            {"id": 0, "type": "ROOT", "pos": [0.0, 0.0, 0.0]},
            {"id": 1, "type": "JUNCTION", "pos": [0.0, 0.0, 0.04]},
            {"id": 2, "type": "STALK_TIP", "pos": [0.015, 0.0, 0.05]},
            {"id": 3, "type": "LEAF_TIP", "pos": [0.03, 0.0, 0.06]},
        ],
        "edges": [
            {
                "source": 0,
                "target": 1,
                "type": "STEM",
                "path": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.04]],
            },
            {
                "source": 1,
                "target": 2,
                "type": "STALK",
                "path": [[0.0, 0.0, 0.04], [0.015, 0.0, 0.05]],
            },
            {
                "source": 2,
                "target": 3,
                "type": "CL",
                "path": [[0.015, 0.0, 0.05], [0.03, 0.0, 0.06]],
            },
        ],
    }
    graph_path = cultivar / "CA_Graphs" / "CA_01012025_annotated_ext_graph.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    # This point cloud has no annotation or graph and must be reported, not converted.
    _write_ply(cultivar / "CA_R_PC" / "CA_R_01022025.ply", rows)


def _pheno_raw_from_oriented(xyz_m: np.ndarray) -> np.ndarray:
    result = np.empty_like(xyz_m)
    result[:, 0] = xyz_m[:, 0] * 1000.0 - 50.0
    result[:, 1] = xyz_m[:, 1] * 1000.0 - 740.0
    result[:, 2] = xyz_m[:, 2] * 1000.0
    return result


def create_pheno4d_fixture(root: Path) -> None:
    plant = root / "Tomato01"
    plant.mkdir(parents=True)
    rng = np.random.default_rng(7)
    stem_z = np.linspace(0.0, 0.10, 80)
    stem = np.stack(
        [rng.normal(0.0, 0.001, 80), rng.normal(0.0, 0.001, 80), stem_z], axis=1
    )
    leaf_x = np.linspace(0.0, 0.065, 70)
    leaf = np.stack(
        [leaf_x, rng.normal(0.0, 0.001, 70), 0.055 + 0.15 * leaf_x], axis=1
    )
    soil = np.stack(
        [rng.uniform(-0.03, 0.03, 30), rng.uniform(-0.03, 0.03, 30), np.zeros(30)],
        axis=1,
    )
    xyz = np.concatenate([soil, stem, leaf])
    labels = np.concatenate([np.zeros(30), np.ones(80), np.full(70, 2)])
    values = np.concatenate([_pheno_raw_from_oriented(xyz), labels[:, None]], axis=1)
    lines = [" ".join(f"{value:.6f}" for value in row) for row in values]
    (plant / "T01_0305_a.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    raw_lines = [" ".join(f"{value:.6f}" for value in row[:3]) for row in values]
    (plant / "T01_0306.txt").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    maize = root / "Maize01"
    maize.mkdir()
    (maize / "M01_0305_a.txt").write_text("0 0 0 0 0\n", encoding="utf-8")


class OtherDatasetConversionTests(unittest.TestCase):
    def test_pheno4d_known_swapped_labels_are_corrected_before_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "raw"
            create_pheno4d_fixture(raw_root)
            (raw_root / "Tomato01").rename(raw_root / "Tomato02")
            path = raw_root / "Tomato02/T02_0325_a.txt"
            (path.parent / "T01_0305_a.txt").rename(path)
            values = np.loadtxt(path)
            original_labels = values[:, 3].copy()
            values[original_labels == 0, 3] = 1
            values[original_labels == 1, 3] = 0
            np.savetxt(path, values)
            raw_hash = file_sha256(path)
            cfg = OmegaConf.load(Path(__file__).parents[2] / "configs/data/pheno4d_tomato.yaml")
            cfg.raw_root = str(raw_root)
            cfg.dataset_root = str(root / "dataset")
            cfg.processed_root = cfg.dataset_root
            cfg.split_by_source_plant = {"Tomato02": "test"}
            cfg.expected_complete_instances = 1
            report = preprocess_pheno4d(cfg)
            sample = ProcessedPlantDataset(cfg.dataset_root)[0]
            rows = sample.metadata["point_to_original_index"]
            expected = np.where(original_labels == 0, int(SemanticClass.BACKGROUND),
                                np.where(original_labels == 1, int(SemanticClass.MAIN_STEM),
                                         int(SemanticClass.LEAF)))
            np.testing.assert_array_equal(sample.semantic.numpy(), expected[rows])
            np.testing.assert_array_equal(sample.instance.numpy(),
                                          np.where(original_labels == 0, -1, original_labels)[rows])
            self.assertEqual(sample.metadata["source_label_correction"], {"0": 1, "1": 0})
            self.assertEqual(report["preprocessing_config"]["source_label_corrections"],
                             {"T02_0325_a": {"0": 1, "1": 0}})
            self.assertEqual(file_sha256(path), raw_hash)
            soil = sample.xyz[sample.semantic == int(SemanticClass.BACKGROUND), 2]
            stem = sample.xyz[sample.semantic == int(SemanticClass.MAIN_STEM), 2]
            self.assertLess(float(soil.median()), float(stem.median()))

    def test_pheno4d_raw_z_is_up_and_coordinates_round_trip(self) -> None:
        raw = np.asarray([
            [-50., -740., 0.], [-40., -740., 0.], [-50., -730., 0.],
            [-50., -740., 100.], [-45., -735., 200.],
        ], dtype=np.float32)
        canonical = pheno4d_canonical_orientation(raw)
        np.testing.assert_allclose(canonical[:3, 2], 0.)
        np.testing.assert_allclose(canonical[3], [0., 0., 0.1], atol=1e-7)
        # The old cache's Y axis becomes Z, and its Z axis becomes negative Y.
        old = np.column_stack([raw[:, 0] + 50, raw[:, 2], -raw[:, 1] - 740]) * 0.001
        np.testing.assert_allclose(canonical, old[:, [0, 2, 1]] * [1, -1, 1], atol=1e-7)
        root = canonical[3]
        homogeneous = np.column_stack([canonical - root, np.ones(len(raw))])
        restored = homogeneous @ _normalised_to_source_mm(root).T
        np.testing.assert_allclose(restored[:, :3], raw, atol=2e-5)

    def test_short_near_horizontal_pheno4d_stem_uses_principal_axis(self) -> None:
        along_stem = np.linspace(0.0, 0.015, 80)
        stem = np.stack(
            [
                np.zeros_like(along_stem),
                along_stem,
                np.linspace(0.002, 0.0045, len(along_stem)),
            ],
            axis=1,
        ).astype(np.float32)

        centreline = _stem_centerline(stem, slice_m=0.004)

        self.assertGreaterEqual(len(centreline), 2)
        self.assertLessEqual(float(centreline[0, 2]), float(centreline[-1, 2]))
        self.assertGreater(float(np.ptp(centreline[:, 1])), 0.008)

    def test_complete_tomatopgt_and_annotated_pheno4d_append_flat_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            pgt_root = temp / "TomatoPGT"
            pheno_root = temp / "Pheno4D"
            dataset_root = temp / "dataset"
            create_tomatopgt_fixture(pgt_root)
            create_pheno4d_fixture(pheno_root)
            pgt_cfg = OmegaConf.create(
                {
                    "dataset_id": "tomatopgt",
                    "dataset_name": "TomatoPGT",
                    "dataset_version": "1.0",
                    "raw_root": str(pgt_root),
                    "dataset_root": str(dataset_root),
                    "processed_root": str(dataset_root),
                    "split_by_source_plant": {"CULTIVAR_A": "train"},
                    "graph_aliases": {},
                    "expected_complete_instances": 1,
                    "voxel_size_m": 0.001,
                    "num_points": 100,
                    "max_nodes": 32,
                    "skeleton_spacing_m": 0.006,
                    "visibility_distance_m": 0.01,
                }
            )
            pgt_report = preprocess_tomatopgt(pgt_cfg)
            self.assertEqual(pgt_report["instance_count"], 1)
            self.assertEqual(pgt_report["ignored_incomplete_count"], 1)
            self.assertEqual(pgt_report["next_plant_number"], 2)

            pheno_cfg = OmegaConf.create(
                {
                    "dataset_id": "pheno4d",
                    "dataset_name": "Pheno4D Tomato",
                    "dataset_version": "1",
                    "raw_root": str(pheno_root),
                    "dataset_root": str(dataset_root),
                    "processed_root": str(dataset_root),
                    "split_by_source_plant": {"Tomato01": "val"},
                    "expected_complete_instances": 1,
                    "voxel_size_m": 0.001,
                    "num_points": 180,
                    "max_nodes": 32,
                    "skeleton_spacing_m": 0.006,
                    "visibility_distance_m": 0.01,
                    "normal_neighbors": 10,
                    "stem_slice_m": 0.006,
                    "min_leaf_points": 5,
                    "leaf_skeleton_voxel_m": 0.004,
                    "leaf_graph_neighbors": 6,
                    "leaf_graph_radius_m": 0.02,
                    "leaf_graph_max_points": 500,
                    "reconstructed_graph_confidence": 0.8,
                }
            )
            pheno_report = preprocess_pheno4d(pheno_cfg)
            self.assertEqual(pheno_report["instance_count"], 1)
            self.assertEqual(pheno_report["ignored_incomplete_count"], 1)
            self.assertEqual(pheno_report["next_plant_number"], 3)

            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["datasets"]), {"tomatopgt", "pheno4d"})
            self.assertEqual(
                manifest["datasets"]["tomatopgt"]["ignored_incomplete_count"], 1
            )
            self.assertEqual(
                manifest["datasets"]["pheno4d"]["ignored_incomplete_count"], 1
            )
            self.assertEqual(
                [entry["instance_id"] for entry in manifest["instances"]],
                ["plant_000001", "plant_000002"],
            )
            self.assertEqual(
                pheno_report["preprocessing_config"]["coordinate_frame_version"],
                PHENO4D_COORDINATE_FRAME_VERSION,
            )
            # A cache made under the old frame must be rebuilt even when the raw
            # scan and all user-specified preprocessing settings are unchanged.
            manifest["instances"][1]["preprocessing_hash"] = "retired-y-up-frame"
            (dataset_root / "manifest.json").write_text(json.dumps(manifest))
            refreshed_pheno = preprocess_pheno4d(pheno_cfg)
            self.assertEqual(refreshed_pheno["resumed_instance_count"], 1)
            self.assertEqual(refreshed_pheno["preprocessing_hash"], pheno_report["preprocessing_hash"])
            repeated_pgt = preprocess_tomatopgt(pgt_cfg)
            self.assertEqual(repeated_pgt["new_instance_count"], 0)
            self.assertEqual(repeated_pgt["skipped_instance_count"], 1)
            self.assertEqual(repeated_pgt["next_plant_number"], 3)
            self.assertFalse((dataset_root / "tomatopgt").exists())
            self.assertFalse((dataset_root / "pheno4d").exists())
            for number, config, convert in [
                (1, pgt_cfg, preprocess_tomatopgt), (2, pheno_cfg, preprocess_pheno4d)
            ]:
                partial = dataset_root / f"plant_{number:06d}" / "top_down.npz"
                side = partial.with_name("side.npz")
                self.assertTrue(partial.is_file())
                self.assertTrue(side.is_file())
                full_hash = file_sha256(partial.with_name("sample.npz"))
                partial.unlink()
                side.unlink()
                backfill = convert(config)
                self.assertEqual(backfill["skipped_instance_count"], 1)
                self.assertTrue(partial.is_file())
                self.assertTrue(side.is_file())
                config.top_down = {"occlusion_radius_m": 0.0}
                config.side = {"occlusion_radius_m": 0.0}
                changed = convert(config)
                self.assertEqual(changed["skipped_instance_count"], 1)
                self.assertEqual(changed["preprocessing_hash"], backfill["preprocessing_hash"])
                self.assertEqual(file_sha256(partial.with_name("sample.npz")), full_hash)
                self.assertEqual(
                    changed["instances"][0]["top_down"]["point_count"],
                    changed["instances"][0]["point_count"],
                )
                self.assertEqual(
                    changed["instances"][0]["side"]["point_count"],
                    changed["instances"][0]["point_count"],
                )
            dataset = ProcessedPlantDataset(dataset_root)
            self.assertEqual(len(dataset), 2)

            pgt = dataset[0]
            self.assertEqual(pgt.metadata["skeleton_source"], "source_derived_cloudgraph")
            self.assertTrue(pgt.metadata["rgb_available"])
            self.assertTrue(bool((pgt.semantic == int(SemanticClass.SIDE_STEM)).any()))
            self.assertTrue(bool((pgt.organ_type == int(OrganType.LEAF_STRUCTURE)).any()))
            pgt.graph_target.validate()

            pheno = dataset[1]
            self.assertEqual(pheno.metadata["source_plant_id"], "Tomato01")
            self.assertEqual(pheno.metadata["skeleton_source"], "reconstructed_from_manual_organs")
            self.assertFalse(pheno.metadata["rgb_available"])
            self.assertTrue(bool((pheno.rgb == 0).all()))
            self.assertTrue(bool(np.isfinite(pheno.normals.numpy()).all()))
            self.assertTrue(bool((pheno.semantic == int(SemanticClass.BACKGROUND)).any()))
            self.assertTrue(bool((pheno.semantic == int(SemanticClass.MAIN_STEM)).any()))
            self.assertTrue(bool((pheno.semantic == int(SemanticClass.LEAF)).any()))
            self.assertTrue(bool((pheno.organ_type == int(OrganType.LEAF_STRUCTURE)).any()))
            self.assertTrue(bool(np.allclose(pheno.node_xyz[0].numpy(), 0.0)))
            self.assertEqual(pheno.metadata["coordinate_frame_version"],
                             PHENO4D_COORDINATE_FRAME_VERSION)
            soil = pheno.xyz[pheno.semantic == int(SemanticClass.BACKGROUND)]
            stem = pheno.xyz[pheno.semantic == int(SemanticClass.MAIN_STEM)]
            self.assertLess(float(soil[:, 2].median()), float(stem[:, 2].median()))
            self.assertLess(float(soil[:, 2].max() - soil[:, 2].min()), 1e-6)
            self.assertGreater(float(stem[:, 2].max() - stem[:, 2].min()), 0.09)
            raw_values = np.loadtxt(pheno_root / "Tomato01/T01_0305_a.txt")
            restored = np.column_stack([pheno.xyz.numpy(), np.ones(len(pheno.xyz))]) @ np.asarray(
                pheno.metadata["normalised_to_original"]
            ).T
            np.testing.assert_allclose(
                restored[:, :3], raw_values[pheno.metadata["point_to_original_index"], :3],
                atol=1e-4,
            )
            pheno.graph_target.validate()


if __name__ == "__main__":
    unittest.main()
