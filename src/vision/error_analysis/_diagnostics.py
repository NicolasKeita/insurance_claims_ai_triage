"""Diagnostic event lists: low confidence, localization, masks and wrong parts."""

from vision.error_analysis._overlap import box_iou, polygon_mask_overlap


def _low_confidence(predictions, config):
    return [
        index
        for index, pred in enumerate(predictions)
        if pred.confidence is not None and pred.confidence < config.confidence_threshold
    ]


def _localization_errors(candidates, false_positives, matched_truth, config):
    return [
        {"prediction_index": pred, "truth_index": target, "box_iou": overlap}
        for overlap, pred, target in candidates
        if pred in false_positives and target not in matched_truth and 0.1 <= overlap < config.matching_iou
    ]


def _mask_diagnostics(result, predictions, truth, matches, config):
    if config.image_size is None:
        result["good_predictions"] = [match.copy() for match in matches]
        if config.include_part_errors:
            result["incomplete_masks"] = []
        return
    incomplete_masks = []
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
            result["good_predictions"].append(match.copy())
        else:
            result["segmentation_errors"].append(match.copy())
    if config.include_part_errors:
        result["incomplete_masks"] = incomplete_masks


def _wrong_parts(predictions, truth, false_positives, false_negatives, config):
    return [
        {
            "prediction_index": pred,
            "truth_index": target,
            "box_iou": box_iou(predictions[pred].box, truth[target].box),
            "predicted_class_id": predictions[pred].class_id,
            "truth_class_id": truth[target].class_id,
        }
        for pred in false_positives
        for target in false_negatives
        if predictions[pred].class_id != truth[target].class_id
        and box_iou(predictions[pred].box, truth[target].box) >= config.matching_iou
    ]
