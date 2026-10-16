"""Diagnostic mask failures and held-out smoke selection, without CV weights."""

from types import SimpleNamespace

import pytest

from scripts.check_damage_test_smoke import select_test_entries
from vision.error_analysis import ErrorInstance, analyze_instances, polygon_mask_iou, save_error_examples
from vision.training import DamageTrainingConfig


def test_diagnostic_distinguishes_correct_box_with_bad_mask_from_good_mask():
    left = ((0.1, 0.1), (0.4, 0.1), (0.4, 0.9), (0.1, 0.9))
    right = ((0.6, 0.1), (0.9, 0.1), (0.9, 0.9), (0.6, 0.9))
    truth = [ErrorInstance(0, (0.1, 0.1, 0.9, 0.9), polygon=left)]
    bad = analyze_instances(
        [ErrorInstance(0, truth[0].box, 0.9, right)],
        truth,
        confidence_threshold=0.25,
        image_size=(100, 100),
    )
    assert bad["matches"][0]["box_iou"] == 1.0
    assert bad["segmentation_errors"][0]["polygon_mask_iou"] == 0.0
    assert bad["good_predictions"] == []
    assert bad["false_positives"] == bad["false_negatives"] == []
    good = analyze_instances(
        [ErrorInstance(0, truth[0].box, 0.9, left)],
        truth,
        confidence_threshold=0.25,
        image_size=(100, 100),
    )
    assert good["good_predictions"][0]["polygon_mask_iou"] == 1.0
    assert good["segmentation_errors"] == []


def test_missing_predicted_polygon_is_a_segmentation_failure():
    polygon = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9))
    result = analyze_instances(
        [ErrorInstance(0, (0.1, 0.1, 0.9, 0.9), 0.9)],
        [ErrorInstance(0, (0.1, 0.1, 0.9, 0.9), polygon=polygon)],
        confidence_threshold=0.25,
        image_size=(20, 20),
    )
    assert result["segmentation_errors"][0]["polygon_mask_iou"] == 0
    assert result["good_predictions"] == []
    with pytest.raises(ValueError, match="positive"):
        polygon_mask_iou(polygon, polygon, (0, 20))


def test_smoke_inventory_covers_classes_deduplicates_and_bounds_selection():
    manifest = {
        "category_mapping": [{"yolo_class_id": index} for index in range(3)],
        "prepared_files": {
            "test": [
                {"image": "test/z.jpg", "class_counts": {"2": 1}},
                {"image": "test/a.jpg", "class_counts": {"0": 1, "1": 1}},
                {"image": "test/b.jpg", "class_counts": {"0": 1}},
                {"image": "test/d.jpg", "class_counts": {"1": 1}},
            ]
        },
    }
    selected = select_test_entries(
        manifest,
        maximum_images=3,
        diagnostic_examples=[{"filename": "test/a.jpg"}, {"filename": "test/d.jpg"}],
    )
    assert [entry["image"] for entry in selected] == ["test/a.jpg", "test/z.jpg", "test/d.jpg"]
    with pytest.raises(ValueError, match="absent from the frozen TEST"):
        select_test_entries(manifest, maximum_images=3, diagnostic_examples=[{"filename": "train/a.jpg"}])
    with pytest.raises(ValueError, match="positive"):
        select_test_entries(manifest, maximum_images=0)


def test_error_overlay_quotas_select_distinct_good_and_bad_images(tmp_path):
    import json

    from PIL import Image

    root = tmp_path / "prepared"
    root.mkdir()
    entries = []
    for name in ("a", "b", "c"):
        Image.new("RGB", (20, 20)).save(root / f"{name}.jpg")
        (root / f"{name}.txt").write_text(
            "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n2 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n",
            encoding="utf-8",
        )
        entries.append({"image": f"{name}.jpg", "label": f"{name}.txt"})
    config = DamageTrainingConfig(
        version="fixture",
        epochs=100,
        imgsz=640,
        batch=1,
        patience=20,
        seed=42,
        device="cpu",
        workers=0,
        optimizer="auto",
        base_model="yolo26n-seg.pt",
        dataset_yaml=root / "dataset.yaml",
        output_dir=tmp_path / "run",
        experiment_name="fixture",
    )
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxyn=[[0.1, 0.1, 0.4, 0.4], [0.6, 0.1, 0.9, 0.4], [0.1, 0.6, 0.4, 0.9]],
            cls=[0, 1, 0],
            conf=[0.9, 0.8, 0.1],
        ),
        masks=SimpleNamespace(
            xyn=[
                [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
                [[0.6, 0.1], [0.9, 0.1], [0.9, 0.4], [0.6, 0.4]],
                [[0.1, 0.6], [0.4, 0.6], [0.4, 0.9], [0.1, 0.9]],
            ]
        ),
    )
    manifest = {
        "category_mapping": [
            {"yolo_class_id": index, "name": name} for index, name in enumerate(("dent", "scratch", "crack"))
        ],
        "prepared_files": {"test": entries},
    }
    summary = save_error_examples(SimpleNamespace(predict=lambda **_: [result]), config, manifest, 0.25, max_examples=8)
    metadata = json.loads((config.output_dir / "error_examples/metadata.json").read_text(encoding="utf-8"))
    assert summary["examples_saved"] == 3
    assert [row["kind"] for row in metadata["examples"]] == ["good_predictions", "false_positives", "false_negatives"]
    assert len({row["filename"] for row in metadata["examples"]}) == 3
