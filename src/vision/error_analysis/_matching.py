"""Greedy same-class box matching and optional mask diagnostics."""

from typing import Any

from vision.error_analysis._diagnostics import (
    _localization_errors,
    _low_confidence,
    _mask_diagnostics,
    _wrong_parts,
)
from vision.error_analysis._labels import _label_instances, _prediction_instances
from vision.error_analysis._models import AnalysisConfig, ErrorInstance
from vision.error_analysis._overlap import box_iou


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
    accepted = _accepted(predictions, config)
    candidates = _candidates(predictions, truth, accepted)
    matched_pred, matched_truth, matches = _greedy(candidates, predictions, config)
    false_positives = [index for index in accepted if index not in matched_pred]
    false_negatives = [index for index in range(len(truth)) if index not in matched_truth]
    result: dict[str, Any] = {
        "matches": matches,
        "good_predictions": [],
        "segmentation_errors": [],
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "low_confidence": _low_confidence(predictions, config),
        "localization_errors": _localization_errors(candidates, false_positives, matched_truth, config),
    }
    _mask_diagnostics(result, predictions, truth, matches, config)
    if config.include_part_errors:
        result["wrong_parts"] = _wrong_parts(predictions, truth, false_positives, false_negatives, config)
    return result


def _accepted(predictions, config):
    return [
        index
        for index, pred in enumerate(predictions)
        if pred.confidence is not None and pred.confidence >= config.confidence_threshold
    ]


def _candidates(predictions, truth, accepted):
    return [
        (box_iou(predictions[pred].box, target.box), pred, target_index)
        for pred in accepted
        for target_index, target in enumerate(truth)
        if predictions[pred].class_id == target.class_id
    ]


def _greedy(candidates, predictions, config):
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
    return matched_pred, matched_truth, matches


def _inspect_image(model, entry, context):
    from PIL import Image

    image_path = context["root"] / entry["image"]
    predictions = model.predict(
        source=str(image_path),
        conf=context["floor"],
        imgsz=context["config"].imgsz,
        device=context["config"].device,
        verbose=False,
        save=False,
    )
    detected = _prediction_instances(predictions[0])
    truth = _label_instances(context["root"] / entry["label"])
    with Image.open(image_path) as original:
        image_size = original.size
    events = analyze_instances(
        detected,
        truth,
        config=AnalysisConfig(
            confidence_threshold=context["threshold"],
            image_size=image_size,
            include_part_errors=context["examples"].include_part_errors,
        ),
    )
    return events, detected, truth, image_path
