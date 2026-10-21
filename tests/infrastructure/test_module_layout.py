"""Guard the import and filesystem contracts affected by package extraction."""

import subprocess
import sys
from pathlib import Path

from tests.integration_support import PROJECT_ROOT
from vision.cardd_hf_preparation import (
    DEFAULT_HF_MANIFEST_PATH,
    DEFAULT_HF_PREPARED_DIR,
)
from vision.cardd_hf_split import DEFAULT_SPLIT_MANIFEST_PATH
from vision.carparts_dataset import DEFAULT_DATASET_ROOT
from vision.dataset import DEFAULT_MANIFEST_PATH, DEFAULT_PREPARED_DIR, ROOT

REPOSITORY = Path(__file__).resolve().parents[2]


def test_default_paths_still_use_repository_data_and_artifacts():
    assert ROOT == PROJECT_ROOT == REPOSITORY
    assert DEFAULT_PREPARED_DIR == REPOSITORY / "data/external_prepared/cardd_yolo"
    assert DEFAULT_HF_PREPARED_DIR == REPOSITORY / "data/external_prepared/cardd_hf_v1"
    assert DEFAULT_DATASET_ROOT == REPOSITORY / "data/external/carparts-seg"
    assert DEFAULT_MANIFEST_PATH == REPOSITORY / "artifacts/vision/cardd_dataset_manifest.json"
    assert DEFAULT_HF_MANIFEST_PATH == REPOSITORY / "artifacts/vision/cardd_hf_dataset_manifest.json"
    assert DEFAULT_SPLIT_MANIFEST_PATH == REPOSITORY / "artifacts/vision/cardd_hf_split_v2.json"


def test_public_entry_points_import_without_loading_vision_frameworks():
    script = """
import sys
import agent.graph
import agent.models
import agent.planner
import agent.service
import agent.tools
import vision.cardd_hf
import vision.cardd_hf_preparation
import vision.cardd_hf_split
import vision.carparts_dataset
import vision.dataset
import vision.error_analysis
import vision.evaluation
assert not {'torch', 'ultralytics', 'fiftyone'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
