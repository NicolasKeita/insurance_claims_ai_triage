"""Local filesystem prerequisites and process fakes with no service startup."""

import subprocess
from types import SimpleNamespace

from scripts.review_console import configuration as settings
from scripts.review_console import runtime


def configured(monkeypatch, tmp_path):
    demo = tmp_path / "artifacts/Demonstrations/review_demo"
    demo.mkdir(parents=True)
    (demo / "manifest.json").write_text("{}", encoding="utf-8")
    next_cli = tmp_path / "frontend/node_modules/next/dist/bin/next"
    next_cli.parent.mkdir(parents=True)
    next_cli.write_text("// controlled CLI placeholder", encoding="utf-8")
    frontend_runner = tmp_path / "frontend/tooling/run-next.mjs"
    frontend_runner.parent.mkdir(parents=True)
    frontend_runner.write_text("// controlled Next.js runner", encoding="utf-8")
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    monkeypatch.setattr(settings.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(settings.shutil, "which", lambda _: "node-local")
    monkeypatch.setattr(settings.subprocess, "check_output", lambda *args, **kwargs: "v24.10.0\n")
    monkeypatch.setattr(settings.socket, "create_connection", disconnected)
    monkeypatch.delenv("CLAIMS_REVIEW_DEMO_DIR", raising=False)
    return SimpleNamespace(demo_dir=None, frontend_port=5173), demo


def disconnected(*args, **kwargs):
    raise ConnectionRefusedError("No listener")


class Child:
    def __init__(self, pid=1001, *, stubborn=False):
        self.pid, self.stubborn, self.code, self.signals = pid, stubborn, None, []

    def poll(self):
        return self.code

    def send_signal(self, event):
        self.signals.append(event)
        if not self.stubborn:
            self.code = 0

    def wait(self, timeout):
        if self.code is None:
            raise subprocess.TimeoutExpired("controlled child", timeout)
        return self.code


def controlled_run(monkeypatch):
    children, stopped = [], []
    jobs = [(["controlled-api"], ".", "http://localhost/catalog"), (["controlled-next"], ".", "http://localhost/")]

    def spawn(command, **kwargs):
        child = Child(1001 + len(children))
        children.append(child)
        return child

    monkeypatch.setattr(runtime, "configuration", lambda _: ({"CLAIMS_REVIEW_DEMO": "1"}, jobs))
    monkeypatch.setattr(runtime.subprocess, "Popen", spawn)
    monkeypatch.setattr(runtime, "wait_ready", lambda *args: None)
    monkeypatch.setattr(runtime, "stop", stopped.append)
    return children, stopped
