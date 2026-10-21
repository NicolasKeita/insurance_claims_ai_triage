"""Test cache."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vision.cardd_hf import (
    _official_load,
    _PinnedHub,
    default_hf_cache_dir,
)


def test_failed_official_load_removes_only_its_new_db_entry_and_restores_config(tmp_path, monkeypatch):
    original_hub = SimpleNamespace(marker="original")
    fo = ModuleType("fiftyone")
    utils = ModuleType("fiftyone.utils")
    hub = ModuleType("fiftyone.utils.huggingface")
    entries, deleted = set(), []
    fo.config = SimpleNamespace(default_dataset_dir="unrelated_original_directory")
    fo.dataset_exists = lambda name: name in entries

    def delete(name):
        deleted.append(name)
        entries.remove(name)

    fo.delete_dataset = delete
    hub.hfh = original_hub

    def failing(repo, **kwargs):
        entries.add(kwargs["name"])
        assert isinstance(hub.hfh, _PinnedHub)
        assert fo.config.default_dataset_dir.startswith(str(tmp_path))
        raise RuntimeError("synthetic interrupted download")

    hub.load_from_hub = failing
    fo.utils, utils.huggingface = utils, hub
    for name, module in [("fiftyone", fo), ("fiftyone.utils", utils), ("fiftyone.utils.huggingface", hub)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("FIFTYONE_DATABASE_DIR", str(tmp_path / "db"))
    with pytest.raises(RuntimeError, match="interrupted download"):
        _official_load("owner/subset", "a" * 40, tmp_path)
    assert len(deleted) == 1 and deleted[0].startswith("cardd_hf_")
    assert not entries
    assert hub.hfh is original_hub
    assert fo.config.default_dataset_dir == "unrelated_original_directory"


def test_external_cache_setting_is_supported(monkeypatch, tmp_path):
    monkeypatch.setenv("CARDD_HF_CACHE_DIR", str(tmp_path / "outside_checkout"))
    assert default_hf_cache_dir() == (tmp_path / "outside_checkout").resolve()


def test_fiftyone_is_not_imported_for_offline_fixtures():
    assert "fiftyone" not in sys.modules
