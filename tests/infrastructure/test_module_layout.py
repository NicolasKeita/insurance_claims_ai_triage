"""Guard the import and filesystem contracts affected by package extraction."""

import subprocess
import sys
from pathlib import Path

from nk_python.structure_checks import discover

from retrieval.knowledge.config import ROOT as KNOWLEDGE_ROOT
from tests.infrastructure.integration_support import PROJECT_ROOT
from vision.dataset import DEFAULT_MANIFEST_PATH, DEFAULT_PREPARED_DIR, ROOT
from vision.dataset.cardd_hf_preparation import (
    DEFAULT_HF_MANIFEST_PATH,
    DEFAULT_HF_PREPARED_DIR,
)
from vision.dataset.cardd_hf_split import DEFAULT_SPLIT_MANIFEST_PATH
from vision.dataset.carparts_dataset import DEFAULT_DATASET_ROOT

REPOSITORY = Path(__file__).resolve().parents[2]


def test_default_paths_still_use_repository_data_and_artifacts():
    assert ROOT == PROJECT_ROOT == KNOWLEDGE_ROOT == REPOSITORY
    assert DEFAULT_PREPARED_DIR == REPOSITORY / "data/external_prepared/cardd_yolo"
    assert DEFAULT_HF_PREPARED_DIR == REPOSITORY / "data/external_prepared/cardd_hf_v1"
    assert DEFAULT_DATASET_ROOT == REPOSITORY / "data/external/carparts-seg"
    assert DEFAULT_MANIFEST_PATH == REPOSITORY / "artifacts/vision/cardd_dataset_manifest.json"
    assert DEFAULT_HF_MANIFEST_PATH == REPOSITORY / "artifacts/vision/cardd_hf_dataset_manifest.json"
    assert DEFAULT_SPLIT_MANIFEST_PATH == REPOSITORY / "artifacts/vision/cardd_hf_split_v2.json"


def test_directory_layout_respects_the_linter_subdirectory_limit():
    _, issues = discover([REPOSITORY / "src", REPOSITORY / "tests"])
    assert not [str(issue) for issue in issues if issue.code == "PY009"]


def test_public_entry_points_import_without_loading_vision_frameworks():
    script = """
import sys
import agent.graph
import agent.models
import agent.planner
import agent.service
import agent.tools
import vision.dataset.cardd_hf
import vision.dataset.cardd_hf_preparation
import vision.dataset.cardd_hf_split
import vision.dataset.carparts_dataset
import vision.dataset
import vision.damage.error_analysis
import vision.damage.evaluation
assert not {'torch', 'ultralytics', 'fiftyone'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
