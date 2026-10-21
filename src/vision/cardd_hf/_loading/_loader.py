"""Loader."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from vision.cardd_hf._constants import _LOAD_LOCK, DEFAULT_HF_REPOSITORY
from vision.cardd_hf._loading._hub import _PinnedHub, _resolve_revision
from vision.cardd_hf._models import CarddHfConfig, HfDamageDataset
from vision.cardd_hf._normalization import normalize_fiftyone_dataset
from vision.dataset import DatasetValidationError


def default_hf_cache_dir() -> Path:
    configured = os.environ.get("CARDD_HF_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    parent = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache")))
    return (parent / "insurance_claims_ai_triage" / "datasets").resolve()


def _official_load(repository: str, revision: str, cache_dir: Path) -> Any:
    """
    This name was absent before this invocation. Remove only its incomplete database entry; source/cache
    files remain resumable.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(cache_dir / "fiftyone_db"))
    fo, fo_hf = _fiftyone_modules()
    name = "cardd_hf_" + hashlib.sha256((repository + "@" + revision).encode()).hexdigest()[:20]
    with _LOAD_LOCK:
        if fo.dataset_exists(name):
            dataset = fo.load_dataset(name)
            if dataset.info.get("source_repo") != repository or dataset.info.get("source_revision") != revision:
                raise DatasetValidationError("Existing FiftyOne dataset source metadata disagrees with its pinned name")
            return dataset
        previous_dir, previous_hub = fo.config.default_dataset_dir, fo_hf.hfh
        fo.config.default_dataset_dir = str(cache_dir / "fiftyone" / revision)
        fo_hf.hfh = _PinnedHub(previous_hub, revision, cache_dir / "huggingface")
        try:
            dataset = fo_hf.load_from_hub(repository, revision=revision, name=name, persistent=True, overwrite=False)
            dataset.info.update(
                source_repo=repository,
                source_revision=revision,
                cardd_hf_source_repository=repository,
                cardd_hf_source_revision=revision,
                source_name=f"CarDD HF mirror ({repository})",
            )
            dataset.save()
            return dataset
        except Exception:
            if fo.dataset_exists(name):
                fo.delete_dataset(name)
            raise
        finally:
            fo.config.default_dataset_dir, fo_hf.hfh = previous_dir, previous_hub


def load_cardd_hf(config: CarddHfConfig | None = None) -> HfDamageDataset:
    """Load the mirror using official ``fiftyone.utils.huggingface.load_from_hub``.

    Dependencies/network/database are lazy. Immutable revisions isolate caches.
    Injected loaders/resolvers support small offline fixtures without MongoDB.
    """
    config = config or CarddHfConfig()
    repository = config.repository or os.environ.get("CARDD_HF_DATASET", DEFAULT_HF_REPOSITORY)
    if not isinstance(repository, str) or not repository.strip():
        raise DatasetValidationError("CARDD_HF_DATASET must identify a Hugging Face dataset repository")
    revision = config.revision if config.revision is not None else os.environ.get("CARDD_HF_REVISION")
    resolved = (config.revision_resolver or _resolve_revision)(repository, revision)
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", resolved):
        raise DatasetValidationError("Dataset revision resolver must return a 40-character commit SHA")
    cache = Path(config.cache_dir).expanduser().resolve() if config.cache_dir is not None else default_hf_cache_dir()
    try:
        raw = (
            _official_load(repository, resolved.lower(), cache)
            if config.loader is None
            else config.loader(repository, revision=resolved.lower())
        )
    except DatasetValidationError:
        raise
    except Exception as exc:
        raise DatasetValidationError(
            f"FiftyOne could not load CarDD HF mirror ({repository}) at {resolved}: {type(exc).__name__}: {exc}"
        ) from exc
    return normalize_fiftyone_dataset(raw, repository, resolved.lower(), annotation_field=config.annotation_field)


def _fiftyone_modules():
    try:
        import fiftyone as fo
        import fiftyone.utils.huggingface as fo_hf
    except ImportError as exc:
        raise DatasetValidationError(
            "FiftyOne is required for this HF export. "
            "Install the project's cardd-hf optional dependencies in a supported Python environment."
        ) from exc
    return fo, fo_hf
