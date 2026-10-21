"""Matching."""

from __future__ import annotations

from typing import Any

from vision.error_analysis._masks import polygon_mask_overlap
from vision.error_analysis._models import AnalysisConfig, ErrorInstance


def box_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    overlap *= max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - overlap
    return overlap / union if union else 0.0


def analyze_instances(
    predictions: list[ErrorInstance],
    truth: list[ErrorInstance],
    *,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Greedily match highest-IoU same-class boxes, once per instance.

    FP/FN are diagnostic approximations at a fixed threshold, not the framework's
    mask AP matching. Optional mask IoU rasterizes the prepared YOLO polygons at
    source image resolution, so it is also separate from framework AP matching.
    """
    accepted, candidates, matched_pred, matched_truth, matches = _match_boxes(predictions, truth, config)
    false_positives = [index for index in accepted if index not in matched_pred]
    good_predictions, segmentation_errors, incomplete_masks = _match_masks(predictions, truth, matches, config)
    result = {
        "matches": matches,
        "good_predictions": good_predictions,
        "segmentation_errors": segmentation_errors,
        "false_positives": false_positives,
        "false_negatives": [index for index in range(len(truth)) if index not in matched_truth],
        "low_confidence": [
            index
            for index, pred in enumerate(predictions)
            if pred.confidence is not None and pred.confidence < config.confidence_threshold
        ],
        "localization_errors": [
            {"prediction_index": pred, "truth_index": target, "box_iou": overlap}
            for overlap, pred, target in candidates
            if pred in false_positives and target not in matched_truth and 0.1 <= overlap < config.matching_iou
        ],
    }
    _part_errors(result, predictions, truth, config, incomplete_masks)
    return result


def _match_boxes(predictions, truth, config):
    accepted = [
        index
        for index, pred in enumerate(predictions)
        if pred.confidence is not None and pred.confidence >= config.confidence_threshold
    ]
    candidates = [
        (box_iou(predictions[pred].box, target.box), pred, target_index)
        for pred in accepted
        for target_index, target in enumerate(truth)
        if predictions[pred].class_id == target.class_id
    ]
    matched_pred, matched_truth, matches = set(), set(), []
    for overlap, pred, target in sorted(
        candidates,
        key=lambda value: (value[0], predictions[value[1]].confidence, -value[1], -value[2]),
        reverse=True,
    ):
        if overlap >= config.matching_iou and pred not in matched_pred and target not in matched_truth:
            matched_pred.add(pred)
            matched_truth.add(target)
            matches.append({"prediction_index": pred, "truth_index": target, "box_iou": overlap})
    return accepted, candidates, matched_pred, matched_truth, matches


def _match_masks(predictions, truth, matches, config):
    good_predictions, segmentation_errors, incomplete_masks = [], [], []
    if config.image_size is not None:
        for match in matches:
            pred = predictions[match["prediction_index"]]
            target = truth[match["truth_index"]]
            overlap = polygon_mask_overlap(pred.polygon, target.polygon, config.image_size)
            match["polygon_mask_iou"] = overlap["iou"]
            if config.include_part_errors:
                match["truth_mask_coverage"] = overlap["truth_coverage"]
                if pred.polygon and target.polygon and overlap["truth_coverage"] < 0.8:
                    incomplete_masks.append(match.copy())
            if match["polygon_mask_iou"] >= config.matching_iou:
                good_predictions.append(match.copy())
            else:
                segmentation_errors.append(match.copy())
    else:
        good_predictions = [match.copy() for match in matches]
    return good_predictions, segmentation_errors, incomplete_masks


def _part_errors(result, predictions, truth, config, incomplete_masks):
    if config.include_part_errors:
        result["wrong_parts"] = [
            {
                "prediction_index": pred,
                "truth_index": target,
                "box_iou": box_iou(predictions[pred].box, truth[target].box),
                "predicted_class_id": predictions[pred].class_id,
                "truth_class_id": truth[target].class_id,
            }
            for pred in result["false_positives"]
            for target in result["false_negatives"]
            if predictions[pred].class_id != truth[target].class_id
            and box_iou(predictions[pred].box, truth[target].box) >= config.matching_iou
        ]
        result["incomplete_masks"] = incomplete_masks
