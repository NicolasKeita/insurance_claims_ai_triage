"""Metadata payload for a bounded error-overlay diagnostic run."""


def build_metadata(context, totals, selected):
    include_parts = context["examples"].include_part_errors
    return {
        "split": "test",
        "images_inspected": len(context["images"]),
        "maximum_images": context["examples"].max_images,
        "selection": "first filename-sorted test images; bounded diagnostic sample, not a representative estimate",
        "confidence_threshold": context["threshold"],
        "diagnostic_candidate_threshold": context["floor"],
        "matching_box_iou": 0.5,
        "diagnostic_polygon_mask_iou": 0.5,
        "incomplete_mask_definition": "Same-class box match with source-resolution polygon intersection/"
        "truth mask area <0.8; fixed diagnostic criterion, not TEST-tuned."
        if include_parts
        else None,
        "wrong_part_definition": "Unmatched prediction/GT pair of different classes with box IoU >=0.5; "
        "plausible classification confusion, not causal proof."
        if include_parts
        else None,
        "counts_in_inspected_sample": totals,
        "good_prediction_definition": "Same-class one-to-one box IoU >=0.5 and source-resolution rasterized "
        "polygon mask IoU >=0.5; other damage in the same image may still be missed.",
        "limitations": "Greedy same-class box IoU FP/FN approximation, not exact mask FP/FN or AP. "
        "Segmentation errors are box matches with prepared-polygon raster IoU <0.5; "
        "rasterization and bridge/hole boundaries differ from framework masks. "
        "Localization errors use unmatched box IoU [0.1,0.5). "
        "Low-confidence predictions below candidate threshold are unavailable. No threshold tuning on TEST.",
        "examples": selected,
    }
