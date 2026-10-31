"""Interruptions and startup failures stop only the children actually launched."""

import pytest

from scripts.review_console import runtime
from tests.api.review.launcher._support import controlled_run


def test_ctrl_c_stops_both_children_in_reverse_order(monkeypatch):
    children, stopped = controlled_run(monkeypatch)

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime.time, "sleep", interrupt)
    assert runtime.run(object()) == 0
    assert len(children) == 2 and stopped == list(reversed(children))


def test_failed_first_server_prevents_frontend_start_and_is_cleaned_up(monkeypatch):
    children, stopped = controlled_run(monkeypatch)

    def not_ready(*args):
        raise RuntimeError("controlled startup failure")

    monkeypatch.setattr(runtime, "wait_ready", not_ready)
    with pytest.raises(RuntimeError, match="startup failure"):
        runtime.run(object())
    assert len(children) == 1 and stopped == children


def test_a_server_exit_shuts_down_the_other_owned_server(monkeypatch):
    children, stopped = controlled_run(monkeypatch)

    def exit_first(_):
        children[0].code = 1

    monkeypatch.setattr(runtime.time, "sleep", exit_first)
    with pytest.raises(RuntimeError, match="A server stopped"):
        runtime.run(object())
    assert stopped == list(reversed(children))


def test_interrupt_during_startup_also_cleans_up_the_started_child(monkeypatch):
    children, stopped = controlled_run(monkeypatch)

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime, "wait_ready", interrupt)
    assert runtime.run(object()) == 0
    assert len(children) == 1 and stopped == children
