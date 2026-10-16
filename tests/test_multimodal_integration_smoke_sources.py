"""The diagnostic runner cannot follow source paths outside its known dataset."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "integration_smoke_script", Path("scripts/check_multimodal_integration_smoke.py")
)
SMOKE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


def manifests(tmp_path, images):
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    yaml = dataset / "dataset.yaml"
    yaml.write_text(json.dumps({"path": str(dataset)}), encoding="utf-8")
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"status": "completed", "config": {"dataset_yaml": str(yaml)}}), encoding="utf-8")
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"images": [{"image": image} for image in images]}), encoding="utf-8")
    return dataset, prior, training


def test_source_escape_rejected_before_any_model_can_be_loaded(tmp_path):
    _, prior, training = manifests(tmp_path, ["../outside.jpg"])
    with pytest.raises(ValueError, match="outside the source dataset"):
        SMOKE.select_sources(prior, training)


def test_prior_selection_order_is_kept_and_missing_images_are_not_silently_dropped(tmp_path):
    """
    Missing files remain selected so the runner records failures explicitly.
    """
    dataset, prior, training = manifests(tmp_path, ["images/test/z.jpg", "images/test/a.jpg", "images/test/z.jpg"])
    actual_root, sources = SMOKE.select_sources(prior, training)
    assert actual_root == dataset
    assert [item.filename for item in sources] == ["test/z.jpg", "test/a.jpg"]
    assert all(not item.path.exists() for item in sources)
