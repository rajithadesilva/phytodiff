"""Canonical Stage 1 metrics used by training, reports, and visualizations."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import Tensor

from tomato_recon.data.collate import collate_plant_samples
from tomato_recon.data.schemas import IGNORE_INDEX, EncoderOutput, PlantBatch, PlantSample, TopologyRole
from tomato_recon.models.encoders.base import encoder_losses, nearest_skeleton_targets
from tomato_recon.train.common import TrainingProgress

SEMANTIC_NAMES = ("background", "leaf", "main_stem", "support_pole", "side_stem")


class EncoderMetricAccumulator:
    def __init__(
        self,
        num_classes: int,
        *,
        skeleton_threshold_m: float = 0.006,
        probability_threshold: float = 0.5,
    ) -> None:
        self.num_classes = num_classes
        self.skeleton_threshold_m = skeleton_threshold_m
        self.probability_threshold = probability_threshold
        self.semantic_intersection = torch.zeros(num_classes, dtype=torch.float64)
        self.semantic_union = torch.zeros(num_classes, dtype=torch.float64)
        self.skeleton_tp = 0
        self.skeleton_predicted = 0
        self.skeleton_target = 0
        self.junction_tp = 0
        self.junction_predicted = 0
        self.junction_target = 0
        self.offset_absolute_error = 0.0
        self.offset_elements = 0
        self.loss_totals: dict[str, float] = {}
        self.loss_weight = 0

    def update(
        self,
        batch: PlantBatch,
        output: EncoderOutput,
        losses: dict[str, Tensor] | None = None,
    ) -> None:
        prediction = output.semantic_logits.argmax(dim=-1)
        labelled = batch.point_valid & (batch.semantic != IGNORE_INDEX)
        for semantic_class in range(self.num_classes):
            predicted_class = prediction == semantic_class
            target_class = batch.semantic == semantic_class
            self.semantic_intersection[semantic_class] += float(
                ((predicted_class & target_class) & labelled).sum()
            )
            self.semantic_union[semantic_class] += float(
                ((predicted_class | target_class) & labelled).sum()
            )

        skeleton_target, offset_target = nearest_skeleton_targets(
            batch.xyz, batch.node_xyz, batch.node_valid, self.skeleton_threshold_m
        )
        skeleton_prediction = (
            output.skeleton_logits.squeeze(-1).sigmoid() >= self.probability_threshold
        )
        valid = batch.point_valid
        self.skeleton_tp += int((skeleton_prediction & skeleton_target & valid).sum())
        self.skeleton_predicted += int((skeleton_prediction & valid).sum())
        self.skeleton_target += int((skeleton_target & valid).sum())

        offset_mask = skeleton_target & valid
        if offset_mask.any():
            self.offset_absolute_error += float(
                (output.centreline_offset[offset_mask] - offset_target[offset_mask]).abs().sum()
            )
            self.offset_elements += int(offset_mask.sum()) * 3

        junction_nodes = batch.node_valid & (
            batch.topology_role == int(TopologyRole.JUNCTION)
        )
        if junction_nodes.any():
            junction_distance = torch.cdist(batch.xyz, batch.node_xyz).masked_fill(
                ~junction_nodes[:, None], torch.inf
            )
            junction_target = (
                junction_distance.min(dim=-1).values
                <= 1.5 * self.skeleton_threshold_m
            )
        else:
            junction_target = torch.zeros_like(valid)
        junction_prediction = (
            output.junction_logits.squeeze(-1).sigmoid() >= self.probability_threshold
        )
        self.junction_tp += int((junction_prediction & junction_target & valid).sum())
        self.junction_predicted += int((junction_prediction & valid).sum())
        self.junction_target += int((junction_target & valid).sum())

        if losses:
            weight = len(batch.plant_ids)
            for name, value in losses.items():
                self.loss_totals[name] = self.loss_totals.get(name, 0.0) + float(value) * weight
            self.loss_weight += weight

    @staticmethod
    def _precision_recall_f1(tp: int, predicted: int, target: int) -> tuple[float, float, float]:
        precision = tp / max(predicted, 1)
        recall = tp / max(target, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        return precision, recall, f1

    def compute(self) -> dict[str, float]:
        present_ious: list[float] = []
        metrics: dict[str, float] = {}
        for index in range(self.num_classes):
            union = float(self.semantic_union[index])
            iou = float(self.semantic_intersection[index] / union) if union else 0.0
            name = SEMANTIC_NAMES[index] if index < len(SEMANTIC_NAMES) else str(index)
            metrics[f"semantic_iou_{name}"] = iou
            if union:
                present_ious.append(iou)
        metrics["semantic_miou"] = sum(present_ious) / max(len(present_ious), 1)

        skeleton_precision, skeleton_recall, skeleton_f1 = self._precision_recall_f1(
            self.skeleton_tp, self.skeleton_predicted, self.skeleton_target
        )
        junction_precision, junction_recall, junction_f1 = self._precision_recall_f1(
            self.junction_tp, self.junction_predicted, self.junction_target
        )
        offset_mae = self.offset_absolute_error / max(self.offset_elements, 1)
        offset_score = math.exp(-offset_mae / 0.006)
        metrics.update(
            {
                "skeleton_precision": skeleton_precision,
                "skeleton_recall": skeleton_recall,
                "skeleton_f1": skeleton_f1,
                "centreline_offset_mae_m": offset_mae,
                "centreline_offset_mae_mm": offset_mae * 1000.0,
                "centreline_offset_score": offset_score,
                "junction_precision": junction_precision,
                "junction_recall": junction_recall,
                "junction_f1": junction_f1,
            }
        )
        metrics["overall_score"] = (
            0.20 * metrics["semantic_miou"]
            + 0.45 * metrics["skeleton_f1"]
            + 0.25 * metrics["centreline_offset_score"]
            + 0.10 * metrics["junction_f1"]
        )
        if self.loss_weight:
            metrics.update(
                {
                    f"{name}_loss" if name != "loss" else "loss": value / self.loss_weight
                    for name, value in self.loss_totals.items()
                }
            )
        return metrics


def evaluate_encoder_model(
    model: torch.nn.Module,
    loader: Iterable[PlantBatch],
    device: torch.device,
    *,
    num_classes: int,
    epoch: int = 0,
    epoch_total: int = 1,
    stage: str = "encoder/val",
    fast_dev_run: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, float]:
    total_batches = len(loader)  # type: ignore[arg-type]
    progress = TrainingProgress(stage, epoch + 1, epoch_total, total_batches)
    accumulator = EncoderMetricAccumulator(num_classes)
    running_losses: dict[str, float] = {}
    steps = 0
    model.eval()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            output = model(
                batch.xyz,
                torch.cat([batch.rgb, batch.normals], dim=-1),
                batch.point_valid,
            )
            losses = encoder_losses(
                output,
                batch.semantic,
                batch.point_valid,
                batch.node_xyz,
                batch.node_valid,
                batch.topology_role,
            )
            accumulator.update(batch, output, losses)
            for name, value in losses.items():
                running_losses[name] = running_losses.get(name, 0.0) + float(value)
            steps += 1
            progress.update(running_losses, steps)
            if progress_callback:
                progress_callback(steps, total_batches)
            if fast_dev_run:
                break
    metrics = accumulator.compute()
    progress.close(metrics)
    return metrics


def encoder_metrics_for_sample(
    sample: PlantSample,
    output: EncoderOutput,
    *,
    skeleton_threshold_m: float = 0.006,
    probability_threshold: float = 0.5,
) -> dict[str, float]:
    batch = collate_plant_samples([sample]).to(output.point_xyz.device)
    losses = encoder_losses(
        output,
        batch.semantic,
        batch.point_valid,
        batch.node_xyz,
        batch.node_valid,
        batch.topology_role,
        skeleton_threshold_m=skeleton_threshold_m,
    )
    accumulator = EncoderMetricAccumulator(
        output.semantic_logits.shape[-1],
        skeleton_threshold_m=skeleton_threshold_m,
        probability_threshold=probability_threshold,
    )
    accumulator.update(batch, output, losses)
    return accumulator.compute()
