"""Readiness uses safe reads; Windows fallback targets the owned process tree."""

from contextlib import nullcontext
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from scripts.review_console import runtime
from tests.api.review.launcher._support import Child


def test_readiness_accepts_http_success_without_loading_models(monkeypatch):
    requested = []

    def available(url, **kwargs):
        requested.append(url)
        return nullcontext(SimpleNamespace(status=200))

    monkeypatch.setattr(runtime, "urlopen", available)
    url = "http://127.0.0.1:8000/v1/review/demo-claims"
    runtime.wait_ready(url, [Child()])
    assert requested == [url]


def test_snapshot_service_failure_is_explicit_without_waiting_the_full_timeout(monkeypatch):
    def unavailable(*args, **kwargs):
        raise HTTPError("http://localhost/catalog", 503, "Not configured", None, None)

    monkeypatch.setattr(runtime, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        runtime.wait_ready("http://localhost/catalog", [Child()])


def test_readiness_connection_failure_times_out_without_real_waiting(monkeypatch):
    clock = iter([0, 0.1, 61])

    def unavailable(*args, **kwargs):
        raise URLError("controlled refused connection")

    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    monkeypatch.setattr(runtime, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="not ready after 60s"):
        runtime.wait_ready("http://localhost/catalog", [Child()])


def test_exited_child_is_not_signalled(monkeypatch):
    child = Child()
    child.code = 0
    runtime.stop(child)
    assert not child.signals


def test_normal_shutdown_signals_only_the_owned_process_group(monkeypatch):
    child, groups = Child(), []

    def terminate(pid, event):
        groups.append((pid, event))
        child.code = 0

    monkeypatch.setattr(runtime.os, "killpg", terminate, raising=False)
    runtime.stop(child)
    if runtime.os.name == "nt":
        assert child.signals == [runtime.signal.CTRL_BREAK_EVENT]
    else:
        assert groups == [(child.pid, runtime.signal.SIGTERM)]


@pytest.mark.skipif(runtime.os.name != "nt", reason="Windows process-tree fallback")
def test_windows_timeout_stops_owned_venv_wrapper_and_descendants(monkeypatch):
    child, commands = Child(stubborn=True), []

    def taskkill(command, **kwargs):
        commands.append(command)
        child.code = 0

    monkeypatch.setattr(runtime.subprocess, "run", taskkill)
    runtime.stop(child)
    assert commands == [["taskkill", "/PID", str(child.pid), "/T", "/F"]]
