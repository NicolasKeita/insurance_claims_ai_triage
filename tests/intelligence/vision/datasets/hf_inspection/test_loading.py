"""Test loading."""

from types import SimpleNamespace

import pytest

from tests.intelligence.vision.datasets.hf_inspection._support import fixture
from vision.dataset import DatasetValidationError
from vision.dataset.cardd_hf import (
    CarddHfConfig,
    _PinnedHub,
    load_cardd_hf,
)


def test_loader_uses_env_source_and_pinned_revision_offline(tmp_path, monkeypatch):
    raw, _, _ = fixture(tmp_path)
    monkeypatch.setenv("CARDD_HF_DATASET", "owner/subset")
    monkeypatch.setenv("CARDD_HF_REVISION", "tag")
    calls = []

    def resolve(repo, revision):
        calls.append(("resolve", repo, revision))
        return "b" * 40

    def load(repo, **kwargs):
        calls.append(("load", repo, kwargs))
        return raw

    result = load_cardd_hf(
        CarddHfConfig(
            cache_dir=tmp_path / "cache",
            loader=load,
            revision_resolver=resolve,
        )
    )
    assert result.revision == "b" * 40
    assert calls == [("resolve", "owner/subset", "tag"), ("load", "owner/subset", {"revision": "b" * 40})]
    assert not (tmp_path / "cache").exists()


def test_pin_facade_forwards_sha_cache_and_rejects_wrong_revision(tmp_path):
    calls = []
    delegate = SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs), marker="original")
    proxy = _PinnedHub(delegate, "b" * 40, tmp_path)
    proxy.snapshot_download(repo_id="owner/subset", repo_type="dataset")
    assert calls == [
        {"repo_id": "owner/subset", "repo_type": "dataset", "revision": "b" * 40, "cache_dir": str(tmp_path)}
    ]
    assert proxy.marker == "original"
    with pytest.raises(DatasetValidationError, match="different HF revision"):
        proxy.snapshot_download(revision="c" * 40)
