"""Hub."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from vision.dataset import DatasetValidationError


class _PinnedHub:
    """Keep official FiftyOne snapshot calls pinned even on versions dropping SHA."""

    def __init__(self, delegate: Any, revision: str, cache: Path):
        self.delegate, self.revision, self.cache = delegate, revision, cache

    def __getattr__(self, name: str) -> Any:
        target = getattr(self.delegate, name)
        if name not in ("snapshot_download", "hf_hub_download"):
            return target

        def pinned(*args, **kwargs):
            if kwargs.get("revision") not in (None, self.revision):
                raise DatasetValidationError("FiftyOne attempted to download a different HF revision")
            kwargs.update(revision=self.revision, cache_dir=str(self.cache))
            return target(*args, **kwargs)

        return pinned


def _resolve_revision(repository: str, revision: str | None) -> str:
    if revision is not None and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision.lower()
    try:
        from huggingface_hub import HfApi

        resolved = HfApi().dataset_info(repository, revision=revision).sha
    except Exception as exc:
        raise DatasetValidationError(f"Cannot resolve Hugging Face source revision for {repository}: {exc}") from exc
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", resolved):
        raise DatasetValidationError("Hugging Face did not return an immutable dataset commit SHA")
    return resolved.lower()
