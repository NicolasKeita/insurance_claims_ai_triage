"""Local file and reporting contracts without loading a model or running E2E."""

import json
from dataclasses import replace

import pytest

from evaluation.assertions import evaluate_assertions
from evaluation.models import ComponentResult, PipelineObservation
from evaluation.preflight import prepare_preflight, run_preflight
from evaluation.preflight.files import embedding_files, scenario_files, trained_model_files
from tests.infrastructure.prerequisites._evaluation import gate as gate
from tests.infrastructure.prerequisites._evaluation import ready_gate as ready_gate


def cache(tmp_path):
    for name in ("modules.json", "config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (tmp_path / name).write_text("[]" if name == "modules.json" else "{}", encoding="utf-8")
    return tmp_path


def test_embedding_file_checks_accept_existing_local_model(tmp_path):
    assert embedding_files(str(cache(tmp_path)))["version"] == str(tmp_path)


@pytest.mark.parametrize("filename", ["tokenizer.json", "model.safetensors", "config.json"])
def test_missing_embedding_files_are_detected_without_loading(tmp_path, filename):
    model = cache(tmp_path)
    (model / filename).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        embedding_files(str(model))


def test_embedding_cache_lookups_never_download(tmp_path, monkeypatch):
    import huggingface_hub

    seen = []

    def lookup(model, filename):
        seen.append(filename)
        return str(tmp_path / filename)

    cache(tmp_path)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lookup)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("Unexpected download"))
    embedding_files("organization/existing-model")
    assert "model.safetensors" in seen


def test_missing_shard_is_detected(tmp_path):
    model = cache(tmp_path)
    (model / "model.safetensors").unlink()
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "missing.safetensors"}}))
    with pytest.raises(RuntimeError, match="incomplete"):
        embedding_files(str(model))


def test_existing_checkpoint_does_not_require_mlflow(gate):
    config, _ = gate
    assert trained_model_files("triage", config.triage_checkpoint)["checkpoint"] == str(config.triage_checkpoint)


def test_required_reference_document_checked_before_parsing(gate):
    _, scenario = gate
    source = scenario.claim_source
    from pathlib import Path

    Path(source).write_text(json.dumps({"documents": [{"filename": "absent.pdf", "available": True}]}))
    scenario = scenario.model_copy(update={"inputs": {"use_reference_case": True}})
    with pytest.raises(RuntimeError, match="absent.pdf"):
        scenario_files(scenario)


def test_explicit_failure_policy_and_skipped_assertions_remain_distinct(gate):
    config, scenario = gate
    report = run_preflight(replace(config, dependency_unavailable="FAIL"), scenarios=[scenario])
    assert not report.available("prerequisites:controlled")
    for status in ("SKIP", "FAIL"):
        observation = PipelineObservation(
            components={"prerequisites": ComponentResult(status=status, reason="Missing DB")}
        )
        checks = evaluate_assertions(scenario, observation, mode="real")
        assert len(checks) == 1 and checks[0].status == status
        assert checks[0].message == "Missing DB"


@pytest.mark.parametrize("content", [None, "invalid toml = [", 'benchmark_version = "unknown"'])
def test_missing_or_invalid_configuration_skips_without_service_checks(gate, tmp_path, content, monkeypatch):
    _, scenario = gate
    path = tmp_path / "evaluation.toml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr("evaluation.preflight.run_preflight", lambda *a, **k: pytest.fail("Unexpected service check"))
    config, report = prepare_preflight(path, [scenario])
    assert config.dependency_unavailable == "SKIP"
    assert report.rows[0]["status"] == "SKIP"
    assert "configuration:" in report.blocked_scenarios[scenario.scenario_id][0]


def test_failed_model_loading_blocks_execution_and_updates_eligibility(ready_gate, monkeypatch):
    from evaluation.preflight.session import PreflightSession

    config, scenario = ready_gate
    for name in ("probe_embeddings", "probe_models", "probe_detectors", "probe_mlflow"):
        monkeypatch.setattr(PreflightSession, name, lambda self: None)

    def models(session):
        session.report.rows.extend(
            [
                {"component": "triage", "status": "SKIP", "reason": "Unreadable checkpoint"},
                {"component": "anomaly", "status": "PASS"},
            ]
        )

    monkeypatch.setattr(PreflightSession, "probe_models", models)
    report = run_preflight(config, scenarios=[scenario])
    assert "Unreadable checkpoint" in "; ".join(report.blocked_scenarios[scenario.scenario_id])
    assert not report.available("prerequisites:controlled")
    assert sum(row["component"] == "prerequisites:controlled" for row in report.rows) == 1
