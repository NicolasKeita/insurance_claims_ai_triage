"""Prerequisites, child environment isolation and occupied ports fail safely."""

from contextlib import nullcontext

import pytest

from scripts.review_console import configuration as settings
from tests.api.review.launcher._support import configured


def test_prepared_snapshot_commands_use_the_current_python_and_no_external_stack(monkeypatch, tmp_path):
    options, demo = configured(monkeypatch, tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://unrelated-configuration")
    monkeypatch.setenv("CLAIMS_REVIEW_DEMO", "0")
    env, jobs = settings.configuration(options)
    assert env["CLAIMS_REVIEW_DEMO"] == "1"
    assert env["CLAIMS_REVIEW_DEMO_DIR"] == str(demo.resolve())
    assert "DATABASE_URL" not in env
    assert settings.os.environ["DATABASE_URL"] == "postgresql://unrelated-configuration"
    assert settings.os.environ["CLAIMS_REVIEW_DEMO"] == "0"
    assert jobs[0][0][:4] == [settings.sys.executable, "-m", "uvicorn", "api.main:app"]
    assert jobs[0][2].endswith("/v1/review/demo-claims")
    assert jobs[1][0][0] == "node-local" and "--strictPort" in jobs[1][0]
    assert all("/health" not in url for _, _, url in jobs)


def test_demo_directory_option_takes_precedence_over_environment(monkeypatch, tmp_path):
    options, demo = configured(monkeypatch, tmp_path)
    custom = tmp_path / "saved"
    custom.mkdir()
    (custom / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CLAIMS_REVIEW_DEMO_DIR", str(custom))
    assert settings.configuration(options)[0]["CLAIMS_REVIEW_DEMO_DIR"] == str(custom.resolve())
    options.demo_dir = demo
    assert settings.configuration(options)[0]["CLAIMS_REVIEW_DEMO_DIR"] == str(demo.resolve())


def test_missing_manifest_does_not_prepare_or_overwrite_snapshots(monkeypatch, tmp_path):
    options, _ = configured(monkeypatch, tmp_path)
    options.demo_dir = tmp_path / "not-prepared"
    with pytest.raises(RuntimeError, match="prepare_review_demo.py first"):
        settings.configuration(options)
    assert not options.demo_dir.exists()


@pytest.mark.parametrize("version", ["v20.19.0", "v22.11.0"])
def test_unsupported_node_has_an_explicit_prerequisite_message(monkeypatch, tmp_path, version):
    options, _ = configured(monkeypatch, tmp_path)
    monkeypatch.setattr(settings.subprocess, "check_output", lambda *args, **kwargs: version)
    with pytest.raises(RuntimeError, match="Node.js 22.12"):
        settings.configuration(options)


def test_missing_frontend_dependencies_are_not_installed_automatically(monkeypatch, tmp_path):
    options, _ = configured(monkeypatch, tmp_path)
    (tmp_path / "frontend/node_modules/vite/bin/vite.js").unlink()
    with pytest.raises(RuntimeError, match="npm --prefix frontend ci"):
        settings.configuration(options)


def test_occupied_port_is_reported_without_touching_its_owner(monkeypatch, tmp_path):
    options, _ = configured(monkeypatch, tmp_path)
    monkeypatch.setattr(settings.socket, "create_connection", lambda *args, **kwargs: nullcontext(object()))
    with pytest.raises(RuntimeError, match="Port 8000 is in use"):
        settings.configuration(options)


@pytest.mark.parametrize("port", [0, 8000, 65536])
def test_frontend_port_cannot_conflict_with_the_fixed_api_proxy(monkeypatch, tmp_path, port):
    options, _ = configured(monkeypatch, tmp_path)
    options.frontend_port = port
    with pytest.raises(RuntimeError, match="frontend port"):
        settings.configuration(options)
