"""Test isolation."""

import subprocess
import sys

import pytest

from tests.integration_support import PROJECT_ROOT
from tests.vision.datasets.hf_preparation._support import (
    mirror_fixture,
    prepare_fixture,
)
from vision.cardd_hf_preparation import prepare_cardd_hf_yolo
from vision.dataset import DatasetValidationError


def test_hf_no_overwrite_and_no_source_modification(tmp_path):
    dataset = mirror_fixture(tmp_path)
    before = {sample.filepath: sample.filepath.read_bytes() for sample in dataset.samples}
    _, output, _ = prepare_fixture(tmp_path, dataset)
    assert {sample.filepath: sample.filepath.read_bytes() for sample in dataset.samples} == before
    with pytest.raises(DatasetValidationError, match="already exists"):
        prepare_cardd_hf_yolo(
            dataset, output, split_manifest_path=tmp_path / "split_v1.json", manifest_path=None, converter="native"
        )
    with pytest.raises(DatasetValidationError, match="outside and separate"):
        prepare_cardd_hf_yolo(
            dataset, dataset.samples[0].filepath.parent / "prepared", manifest_path=None, converter="native"
        )


def test_hf_preparation_module_does_not_import_frameworks_or_download():
    code = (
        "import sys; import vision.cardd_hf_preparation; "
        "assert not {'fiftyone', 'ultralytics', 'torch', 'huggingface_hub'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_hf_inspection_cli_requires_explicit_loading_mode():
    script = PROJECT_ROOT / "scripts" / "inspect_cardd_dataset.py"
    result = subprocess.run([sys.executable, str(script)], text=True, capture_output=True)
    assert result.returncode == 2
    assert "--download" in result.stderr and "--dataset-name" in result.stderr
