from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def create_tomatowur_fixture(root: Path, points: int = 80) -> Path:
    point_dir = root / "point_clouds"
    annotation_dir = root / "ann_versions" / "0-paper-2Dto3D" / "annotations" / "fixture_plant"
    split_dir = root / "ann_versions" / "0-paper-2Dto3D" / "json"
    point_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    z = np.linspace(0, 0.2, points)
    x = rng.normal(0, 0.002, points)
    y = rng.normal(0, 0.002, points)
    semantic = np.full(points, 2)
    semantic[points // 2 :] = 4
    semantic[-5:] = 3
    pc_lines = ["x,y,z,blue,green,red,nx,ny,nz"]
    labels = ["semantic,leaf_stem_instances"]
    for index in range(points):
        pc_lines.append(
            f"{x[index]},{y[index]},{z[index]},20,150,30,1,0,0"
        )
        labels.append(f"{semantic[index]},{index // 20}")
    point_path = point_dir / "fixture_plant.csv"
    label_path = annotation_dir / "fixture_plant_labels.csv"
    skeleton_path = annotation_dir / "fixture_plant_skeleton.csv"
    point_path.write_text("\n".join(pc_lines) + "\n", encoding="utf-8")
    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    skeleton_path.write_text(
        "x_skeleton,y_skeleton,z_skeleton,vid,parentid,edgetype,gt_int_length,gt_int_diameter,gt_ph_angle,gt_lf_angle\n"
        "0,0,0,0,,,nan,nan,nan,nan\n"
        "0,0,0.08,1,0,<,0.08,0.006,nan,nan\n"
        "0,0,0.16,2,1,<,0.08,0.005,nan,nan\n"
        "0.05,0,0.11,3,1,+,nan,nan,45,40\n"
        "0.09,0,0.12,4,3,<,nan,nan,nan,nan\n",
        encoding="utf-8",
    )
    split = [
        {
            "plant_id": "fixture_plant",
            "file_name": str(point_path),
            "sem_seg_file_name": str(label_path),
            "skeleton_file_name": str(skeleton_path),
            "genotype": "test_cultivar",
        }
    ]
    split_path = split_dir / "train.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    return split_path

