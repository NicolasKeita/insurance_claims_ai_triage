"""Test diagnostics."""

import json
from types import SimpleNamespace

from tests.intelligence.vision.training.damage._dataset import (
    offline_environment as offline_environment,
)
from tests.intelligence.vision.training.damage._dataset import prepared
from tests.intelligence.vision.training.damage._dataset import prepared as prepared
from vision.damage.error_analysis import (
    AnalysisConfig,
    ErrorExamplesConfig,
    ErrorInstance,
    analyze_instances,
    box_iou,
    save_error_examples,
)


def test_error_matching_is_one_to_one_classwise_and_keeps_low_confidence():
    truth = [ErrorInstance(0, (0, 0, 0.5, 0.5)), ErrorInstance(1, (0.5, 0.5, 1, 1))]
    predictions = [
        ErrorInstance(0, (0, 0, 0.5, 0.5), 0.9),
        ErrorInstance(0, (0, 0, 0.5, 0.5), 0.8),
        ErrorInstance(1, (0.5, 0.5, 1, 1), 0.1),
    ]
    result = analyze_instances(predictions, truth, config=AnalysisConfig(confidence_threshold=0.25))
    assert len(result["matches"]) == 1
    assert result["false_positives"] == [1] and result["false_negatives"] == [1]
    assert result["low_confidence"] == [2]
    assert box_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1


def test_wrong_class_is_fp_and_fn_not_a_match():
    result = analyze_instances(
        [ErrorInstance(1, (0, 0, 1, 1), 0.9)],
        [ErrorInstance(0, (0, 0, 1, 1))],
        config=AnalysisConfig(confidence_threshold=0.25),
    )
    assert result["false_positives"] == [0] and result["false_negatives"] == [0]


def test_error_examples_follow_manifest_nested_paths(prepared):
    from PIL import Image

    config, manifest = prepared
    root = config.dataset_yaml.parent
    image = root / "images/test/nested/tiny.png"
    label = root / "labels/test/nested/tiny.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(image)
    label.write_text("1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")
    manifest["prepared_files"] = {
        "test": [{"image": "images/test/nested/tiny.png", "label": "labels/test/nested/tiny.txt"}]
    }
    paths = []

    def predict(**kwargs):
        paths.append(kwargs["source"])
        return [SimpleNamespace(boxes=None)]

    analysis = save_error_examples(
        SimpleNamespace(predict=predict),
        config,
        manifest,
        0.25,
        examples=ErrorExamplesConfig(max_examples=4),
    )
    assert paths == [str(image)] and analysis["examples_saved"] == 1
    payload = json.loads((config.output_dir / "error_examples/metadata.json").read_text())
    assert payload["examples"][0]["kind"] == "false_negatives"
    assert len(list((config.output_dir / "error_examples").glob("*.jpg"))) == 1
