"""Recover interrupted materialization only when the source bytes are identical."""

import hashlib
from uuid import uuid4

from evaluation.deterministic.document_robustness.transformations import materialize_variant


def materialize_or_reuse(reference, variant, source):
    if not source.exists():
        return materialize_variant(reference, variant, source)
    temporary = source.with_name(f"{source.stem}-{uuid4().hex}.pdf")
    try:
        config = materialize_variant(reference, variant, temporary)
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(temporary.read_bytes()).digest():
            raise ValueError(f"Existing source differs from deterministic reference: {source}")
        return config
    finally:
        temporary.unlink(missing_ok=True)
