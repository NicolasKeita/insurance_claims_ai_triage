"""Test overlays."""

from types import SimpleNamespace

from tests.vision.diagnostics.errors._support import (
    error_example_config,
    error_predictions,
)
from vision.error_analysis import (
    ErrorExamplesConfig,
    save_error_examples,
)


def test_error_overlay_quotas_select_distinct_good_and_bad_images(tmp_path):
    import json

    root, entries, config = error_example_config(tmp_path)
    result = error_predictions()
    manifest = {
        "category_mapping": [
            {"yolo_class_id": index, "name": name} for index, name in enumerate(("dent", "scratch", "crack"))
        ],
        "prepared_files": {"test": entries},
    }
    summary = save_error_examples(
        SimpleNamespace(predict=lambda **_: [result]),
        config,
        manifest,
        0.25,
        examples=ErrorExamplesConfig(max_examples=8),
    )
    metadata = json.loads((config.output_dir / "error_examples/metadata.json").read_text(encoding="utf-8"))
    assert summary["examples_saved"] == 3
    assert [row["kind"] for row in metadata["examples"]] == ["good_predictions", "false_positives", "false_negatives"]
    assert len({row["filename"] for row in metadata["examples"]}) == 3
