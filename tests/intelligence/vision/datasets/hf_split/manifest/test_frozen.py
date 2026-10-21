"""Test frozen."""

import hashlib
import json

import pytest

from tests.intelligence.vision.datasets.hf_split._support import fixture
from vision.dataset.cardd_hf_split import (
    create_or_load_cardd_hf_split,
)


def test_frozen_manifest_is_reused_without_rewrite(tmp_path, monkeypatch):
    import vision.dataset.cardd_hf_split._manifest as module

    dataset = fixture(tmp_path)
    path = tmp_path / "split.json"
    first = create_or_load_cardd_hf_split(dataset, path)
    before = path.read_bytes(), path.stat().st_mtime_ns
    monkeypatch.setattr(module, "_assign", lambda *args: pytest.fail("A frozen split must not be regenerated"))
    assert create_or_load_cardd_hf_split(dataset, path) == first
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_existing_v1_manifest_stays_valid_and_immutable_after_v2_upgrade(tmp_path, monkeypatch):
    import vision.dataset.cardd_hf_split._manifest as module

    dataset = fixture(tmp_path)
    path = tmp_path / "cardd_hf_split_v1.json"
    legacy = create_or_load_cardd_hf_split(dataset, tmp_path / "initial.json")
    legacy["split_version"] = "cardd_hf_split_v1"
    legacy["algorithm"] = "deterministic_multilabel_greedy_grouped_v1"
    stable = {
        "source_fingerprint": legacy["source_fingerprint"],
        "sample_ids": legacy["sample_ids"],
        "split_version": legacy["split_version"],
        "algorithm": legacy["algorithm"],
        "seed": legacy["seed"],
        "ratios": legacy["ratios"],
    }
    legacy["dataset_fingerprint"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(legacy), encoding="utf-8")
    before = path.read_bytes(), path.stat().st_mtime_ns
    monkeypatch.setattr(module, "_assign", lambda *args: pytest.fail("The legacy frozen split must be reused"))
    assert create_or_load_cardd_hf_split(dataset, path) == legacy
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
