"""Classify tests and check only the prerequisites of the selected integrations."""

import pytest

INTEGRATION_MARKERS = ("postgres", "qdrant", "rag", "vision")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Classify before pytest applies the requested marker expression."""
    for item in items:
        if any(item.get_closest_marker(name) for name in INTEGRATION_MARKERS):
            if item.get_closest_marker("integration") is None:
                item.add_marker(pytest.mark.integration)
        if item.get_closest_marker("integration") is not None:
            if item.get_closest_marker("unit") is not None:
                raise pytest.UsageError(
                    f"{item.nodeid} has both unit and integration markers. "
                    "A test must belong to exactly one of these suites."
                )
        else:
            item.add_marker(pytest.mark.unit)


def pytest_collection_finish(session):
    if session.config.option.collectonly or session.testsfailed:
        return
    from integration_prerequisites import PrerequisiteError, check_selected_prerequisites

    markers = {
        name for item in session.items for name in INTEGRATION_MARKERS if item.get_closest_marker(name) is not None
    }
    try:
        check_selected_prerequisites(markers, session.config.rootpath)
    except PrerequisiteError as error:
        raise pytest.UsageError(str(error)) from None
