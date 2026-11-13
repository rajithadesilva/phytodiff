"""Geometry and topology metrics under one-to-one spatial node matching."""

from collections import Counter, defaultdict
from datetime import timedelta
import time

import torch
from scipy.optimize import linear_sum_assignment

from tomato_recon.data.aggfm import from_sample, sample_points
from tomato_recon.models.flow_matching.losses import chamfer
from tomato_recon.models.flow_matching.runtime import predict


def sample_skeleton(xyz, parent, samples=8):
    children = (parent >= 0).nonzero().flatten()
    if not len(children):
        return xyz
    fraction = torch.linspace(0, 1, samples, device=xyz.device)[None, :, None]
    lines = xyz[parent[children], None] * (1 - fraction) + xyz[children, None] * fraction
    return torch.cat((xyz, lines.flatten(0, 1)))


def metrics(prediction, data, *, threshold_m=0.02):
    p, gt = prediction.node_xyz, data.target_xyz
    distances = torch.cdist(p.float(), gt.float())
    distances_cpu = distances.detach().cpu().numpy()
    rows, cols = linear_sum_assignment(distances_cpu)
    threshold = distances_cpu.dtype.type(threshold_m)
    mapping = {int(i): int(j) for i, j in zip(rows, cols) if distances_cpu[i, j] <= threshold}
    # Matching and set operations run on the CPU. Transfer complete arrays once
    # instead of synchronizing CUDA for every scalar used by a Python loop.
    edge_pairs = prediction.edges.detach().cpu().tolist()
    target_parents = data.target_parent.detach().cpu().tolist()
    predicted_parents = prediction.parent_index.detach().cpu().tolist()
    predicted_edges = {(mapping[a], mapping[b]) for a, b in edge_pairs
                       if a in mapping and b in mapping}
    gt_edges = {(parent, child) for child, parent in enumerate(target_parents) if parent >= 0}
    correct = len(predicted_edges & gt_edges)
    gt_root = [node for node, parent in enumerate(target_parents) if parent < 0]
    root_correct = mapping.get(prediction.graph.root_node_id, -1) in gt_root
    pred_degree = Counter(parent for parent in predicted_parents if parent >= 0)
    gt_degree = Counter(parent for parent in target_parents if parent >= 0)
    pred_junction = [node for node, degree in pred_degree.items() if degree >= 2]
    gt_junction = {node for node, degree in gt_degree.items() if degree >= 2}
    junction_correct = sum(mapping.get(i, -1) in gt_junction for i in pred_junction)
    try:
        prediction.graph.validate()
        valid = 1.0
    except ValueError:
        valid = 0.0
    result = {
        "node_chamfer_m": float((distances.min(0).values.mean() + distances.min(1).values.mean()) / 2),
        "skeleton_chamfer_m": float(chamfer(sample_skeleton(p, prediction.parent_index),
                                                         sample_skeleton(gt, data.target_parent))),
        "count_absolute_error": abs(len(p) - len(gt)),
        "edge_precision": correct / max(len(prediction.edges), 1),
        "edge_recall": correct / max(len(gt_edges), 1),
        "parent_accuracy": correct / max(len(gt_edges), 1),
        "junction_precision": junction_correct / max(len(pred_junction), 1),
        "junction_recall": junction_correct / max(len(gt_junction), 1),
        "root_accuracy": float(root_correct),
        "root_distance_m": float((p[prediction.graph.root_node_id] - gt[gt_root[0]]).norm()),
        "tree_validity": valid,
    }
    result["junction_f1"] = 2 * result["junction_precision"] * result["junction_recall"] / max(
        result["junction_precision"] + result["junction_recall"], 1e-8)
    # Labels come from the original scan. They do not imply visibility in every partial view.
    if data.visibility is not None:
        nearest = distances.min(0).values
        for label, name in ((0, "observed"), (1, "partial"), (2, "inferred_unknown")):
            select = data.visibility == label
            if select.any():
                result[f"annotated_{name}_target_recall"] = float((nearest[select] <= threshold_m).float().mean())
                result[f"annotated_{name}_target_distance_m"] = float(nearest[select].mean())
    return result


def aggregate(rows):
    totals = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals[key].append(value)
    return {key: sum(values) / len(values) for key, values in totals.items()}


def _print_sample_progress(label, completed, total, started, previous_width=0):
    elapsed = time.perf_counter() - started
    eta = timedelta(seconds=int(elapsed / completed * (total - completed))) if completed else "--:--:--"
    message = (f"{label} | Samples {completed}/{total} ({completed / total:.1%}) | "
               f"elapsed {timedelta(seconds=int(elapsed))} | ETA {eta}")
    padding = " " * max(previous_width - len(message), 0)
    print(f"\r{message}{padding}", end="\n" if completed == total else "", flush=True)
    return len(message)


@torch.no_grad()
def evaluate_dataset(model, encoder, cfg, dataset, *, steps=None, max_samples=None,
                     progress_label=None):
    rows = []
    device = next(model.parameters()).device
    sample_total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    if sample_total <= 0:
        raise ValueError("selected evaluation split/views contain no samples")
    evaluation_started = time.perf_counter()
    progress_width = 0
    if progress_label is not None:
        progress_width = _print_sample_progress(
            progress_label, 0, sample_total, evaluation_started, progress_width)
    for index in range(sample_total):
        data = sample_points(from_sample(dataset[index]), int(cfg.model.aggfm.max_points),
                             int(cfg.seed) + index).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction, _ = predict(model, encoder, data.points_only(), cfg, steps=steps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        runtime = time.perf_counter() - started
        row = {"plant_id": data.plant_id, "dataset": data.dataset, "view": data.view,
               **metrics(prediction, data), "inference_seconds": runtime}
        rows.append(row)
        if progress_label is not None:
            progress_width = _print_sample_progress(
                progress_label, len(rows), sample_total, evaluation_started, progress_width)
    groups = defaultdict(list)
    for row in rows:
        groups[f'{row["dataset"]}/{row["view"]}'].append(row)
    return {"overall": aggregate(rows), "by_source_view": {key: aggregate(items) for key, items in groups.items()},
            "sample_count": len(rows), "samples": rows,
            "visibility_basis": "original full-scan cached annotations; partial-view visibility is not relabelled"}
