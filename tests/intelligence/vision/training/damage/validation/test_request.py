"""Check-only requests must validate inputs before loading a model or writing output."""

import pytest

from tests.intelligence.vision.training.damage._dataset import offline_environment as offline_environment
from tests.intelligence.vision.training.damage._dataset import prepared as prepared
from tests.intelligence.vision.training.damage.validation._framework import diagnostic_params
from vision.damage.validation import prepare_validation


@pytest.mark.parametrize("threshold", [True, -0.1, 1.1, float("nan"), float("inf"), "0.25"])
def test_invalid_operating_confidence_fails_before_loading(prepared, tmp_path, threshold):
    config, manifest = prepared
    params = diagnostic_params(manifest, operating_confidence=threshold)
    with pytest.raises(ValueError, match="Operating confidence"):
        prepare_validation(config, output_dir=tmp_path / "diagnostics", params=params)


@pytest.mark.parametrize("size", [True, 0, -1, 640.5])
def test_invalid_image_size_fails_before_loading(prepared, tmp_path, size):
    config, manifest = prepared
    with pytest.raises(ValueError, match="image size"):
        prepare_validation(
            config, output_dir=tmp_path / "diagnostics", params=diagnostic_params(manifest, image_size=size)
        )


@pytest.mark.parametrize("batch", [True, 0, -1, 4.5])
def test_invalid_batch_size_fails_before_loading(prepared, tmp_path, batch):
    config, manifest = prepared
    params = diagnostic_params(manifest, batch_size=batch)
    with pytest.raises(ValueError, match="batch size"):
        prepare_validation(config, output_dir=tmp_path / "diagnostics", params=params)


def test_existing_and_protected_destinations_are_refused(prepared, tmp_path):
    config, manifest = prepared
    existing = tmp_path / "diagnostics"
    existing.mkdir()
    for destination, error in (
        (existing, FileExistsError),
        (config.output_dir / "diagnostics", ValueError),
        (config.dataset_yaml.parent / "diagnostics", ValueError),
    ):
        with pytest.raises(error):
            prepare_validation(config, output_dir=destination, params=diagnostic_params(manifest))
