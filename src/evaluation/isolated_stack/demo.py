"""Ephemeral real demo stack context: one owned schema plus new random collections."""

from contextlib import contextmanager

from evaluation.isolated_stack.postgres import isolated_postgres
from evaluation.isolated_stack.setup import IsolatedDemoSetup


@contextmanager
def isolated_demo_stack(config, preflight, output_dir):
    """Prepare explicitly requested ephemeral real services using existing indexers.

    Only a safe dedicated test database and new randomly named collections are
    mutated. Existing app data and existing collections are never imported or
    overwritten. All created resources are removed on context exit.
    """
    if not config.database_url or not config.qdrant_url:
        raise ValueError("--isolated-demo-stack requires explicit --database-url and --qdrant-url")
    if not preflight.available("postgresql") or not preflight.available("qdrant"):
        raise RuntimeError("Isolated demo setup requires both explicitly selected services to pass preflight")
    if not preflight.available("embeddings"):
        raise RuntimeError("Isolated real indexing requires a locally loadable embedding model")
    setup = IsolatedDemoSetup(config, preflight, output_dir)
    with isolated_postgres(config.database_url) as stack:
        setup.bind_postgres(stack)
        try:
            setup.index()
            setup.update_rows()
            setup.write_audit()
            yield setup
        finally:
            setup.cleanup()
            setup.write_audit()
