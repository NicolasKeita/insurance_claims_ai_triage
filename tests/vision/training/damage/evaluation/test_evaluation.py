"""Test evaluation."""

import copy
import json

import pytest

from tests.vision.training.damage._dataset import (
    offline_environment as offline_environment,
)
from tests.vision.training.damage._dataset import prepared
from tests.vision.training.damage._dataset import prepared as prepared
from tests.vision.training.damage._framework import OrchestrationStub, completed_fixture
from vision.evaluation import (
    EvaluationParams,
    run_evaluation,
)


def test_test_evaluation_uses_only_exact_val_selected_best(prepared):
    config, manifest = prepared
    completed_fixture(config, manifest)
    loaded = []
    stub = OrchestrationStub()

    def load(path):
        loaded.append(path)
        return stub

    result = run_evaluation(
        config,
        params=EvaluationParams(
            model_factory=load,
            environment_provider=lambda: {},
            dataset_loader=lambda _: manifest,
            max_error_examples=0,
        ),
    )
    assert loaded == [str(config.output_dir / "train/weights/best.pt")]
    assert stub.validation_kwargs["split"] == "test" and stub.validation_kwargs["conf"] == 0.25
    assert result["split"] == "test" and result["checkpoint_selection_split"] == "val"
    assert json.loads((config.output_dir / "test_metrics.json").read_text())["mask"]["map50"] == 0.6
    assert (config.output_dir / "per_class_metrics.csv").is_file()
    assert result["per_class"][0]["mask"]["map50"] is None


@pytest.mark.parametrize(
    "change", ["checkpoint_bytes", "test_selected", "alternate_path", "annotation_hash", "manifest_bytes", "yaml_bytes"]
)
def test_evaluation_rejects_provenance_changes_before_model_load(prepared, change):
    config, manifest = prepared
    _, summary = completed_fixture(config, manifest)
    if change == "checkpoint_bytes":
        (config.output_dir / "train/weights/best.pt").write_bytes(b"changed")
    elif change == "test_selected":
        summary["checkpoint_selection_split"] = "test"
    elif change == "alternate_path":
        summary["best_checkpoint"] = "train/weights/last.pt"
    elif change == "annotation_hash":
        manifest = copy.deepcopy(manifest)
        manifest["annotation_hashes"]["test"] = "f" * 64
    elif change == "manifest_bytes":
        (config.dataset_yaml.parent / "cardd_dataset_manifest.json").write_text("{}")
    else:
        config.dataset_yaml.write_text("{}")
    (config.output_dir / "training_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        run_evaluation(
            config,
            params=EvaluationParams(
                dataset_loader=lambda _: manifest,
                model_factory=lambda _: pytest.fail("must not load"),
                environment_provider=lambda: pytest.fail("must not import"),
                max_error_examples=0,
            ),
        )
