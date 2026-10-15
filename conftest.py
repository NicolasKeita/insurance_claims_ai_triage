"""The public pytest interface; this root hook is loaded before collection."""

import pytest


_PANEL_SHOWN = pytest.StashKey[bool]()


@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session):
    config = session.config
    if config.option.collectonly or config.stash.get(_PANEL_SHOWN, False):
        return
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    config.stash[_PANEL_SHOWN] = True
    reporter.write_line("\n================ AVAILABLE TEST SUITES ================\n")
    for line in (
        "pytest              All tests (unit + integration)",
        "pytest -m unit      Unit tests only",
        "pytest -m postgres  PostgreSQL integration tests",
        "pytest -m qdrant    Qdrant integration tests",
        "pytest -m rag       RAG integration tests",
        "pytest -m vision    Vision integration tests",
        "",
        "pytest --markers    Show all available markers",
    ):
        reporter.write_line(line)
    reporter.write_line("\n=======================================================\n")
