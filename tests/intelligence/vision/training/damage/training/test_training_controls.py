"""Optional experiment controls retain the identity of historical runs."""

from dataclasses import replace

import pytest

from tests.intelligence.vision.training.damage._dataset import offline_environment as offline_environment
from tests.intelligence.vision.training.damage._dataset import prepared as prepared
from tests.intelligence.vision.training.damage._framework import completed_fixture
from vision.damage.training import DEFAULT_TRAINING_CONFIG, load_training_config


def test_baseline_payload_keeps_the_historical_configuration_shape():
    config = load_training_config()
    assert config.payload() == {
        "version": "cardd_hf_yolo26n_seg_v1",
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "patience": 20,
        "seed": 42,
        "device": "0",
        "workers": 0,
        "optimizer": "auto",
        "base_model": "yolo26n-seg.pt",
        "dataset_yaml": str(DEFAULT_TRAINING_CONFIG.parents[2] / "data/external_prepared/cardd_hf_v1/dataset.yaml"),
        "output_dir": str(DEFAULT_TRAINING_CONFIG.parents[2] / "artifacts/vision/damage_segmentation_v1"),
        "experiment_name": "insurance-claims-vision-damage",
        "track_mlflow": False,
        "log_checkpoint": False,
    }


def test_baseline_does_not_override_framework_controls(prepared):
    stub, _ = completed_fixture(*prepared)
    assert {"mask_ratio", "mosaic", "scale", "close_mosaic"}.isdisjoint(stub.training_kwargs)


def test_explicit_controls_reach_training_and_recorded_configuration(prepared):
    config, manifest = prepared
    controls = {"mask_ratio": 2, "mosaic": 0.0, "scale": 0.2, "close_mosaic": 10}
    config = replace(config, **controls)
    stub, summary = completed_fixture(config, manifest)
    assert {name: stub.training_kwargs[name] for name in controls} == controls
    assert {name: config.payload()[name] for name in controls} == controls
    assert {name: summary["config"][name] for name in controls} == controls


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("mask_ratio", 0),
        ("mask_ratio", -1),
        ("mask_ratio", 2.0),
        ("mask_ratio", True),
        ("mosaic", -0.1),
        ("mosaic", 1.1),
        ("mosaic", float("nan")),
        ("mosaic", float("inf")),
        ("mosaic", True),
        ("mosaic", "0.5"),
        ("scale", -0.1),
        ("scale", 1.1),
        ("scale", float("nan")),
        ("scale", float("inf")),
        ("scale", False),
        ("scale", "0.5"),
        ("close_mosaic", -1),
        ("close_mosaic", 101),
        ("close_mosaic", 10.0),
        ("close_mosaic", True),
    ],
)
def test_invalid_controls_fail_before_a_run_starts(prepared, name, value):
    with pytest.raises(ValueError, match=name):
        replace(prepared[0], **{name: value})


def test_toml_loads_explicit_controls_and_still_rejects_unknown_keys(tmp_path):
    path = tmp_path / "experiment.toml"
    source = DEFAULT_TRAINING_CONFIG.read_text(encoding="utf-8")
    controls = "\nmask_ratio = 2\nmosaic = 0.0\nscale = 0.2\nclose_mosaic = 10\n"
    path.write_text(source + controls, encoding="utf-8")
    assert load_training_config(path).training_overrides() == {
        "mask_ratio": 2,
        "mosaic": 0.0,
        "scale": 0.2,
        "close_mosaic": 10,
    }
    path.write_text(source + controls + "unsupported_control = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown training configuration keys.*unsupported_control"):
        load_training_config(path)
