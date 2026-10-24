"""Source-image fixture wiring without E2E, services, or model inference."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from evaluation.deterministic.fixtures import materialize_scenario_case
from evaluation.models import EvaluationScenario
from evaluation.real.context import ComponentRecorder, RealRun
from evaluation.real.stages.claim import ClaimSteps
from evaluation.scenarios import load_scenarios

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def visual_source(tmp_path):
    source = tmp_path / "generated.png"
    with Image.new("RGB", (320, 180), "silver") as image:
        image.save(source)
    asset = {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "origin": "ai_generated",
        "annotation": {"damaged_parts": ["HOOD", "FRONT_BUMPER"]},
        "provenance": "Generated source image approved by the user; qualitative part labels only.",
    }
    scenario = EvaluationScenario(
        scenario_id="visual_materialization_contract",
        description="Fixture image-copy contract only",
        claim_source=str(ROOT / "data/CLAIM-2026-00001"),
        vision_required=True,
        external_assets=[asset],
        inputs={"claim_id": "VISUAL-MATERIALIZATION-1", "declared_damage": ["FRONT_BUMPER"]},
    )
    return source, asset, scenario


@pytest.mark.parametrize("origin", ["ai_generated", "photograph"])
def test_real_materialization_copies_source_bytes_and_records_origin(visual_source, tmp_path, monkeypatch, origin):
    source, asset, scenario = visual_source
    asset = {**asset, "origin": origin}
    scenario = scenario.model_copy(update={"external_assets": [asset]})

    def no_geometry(*args, **kwargs):
        pytest.fail("Real fixture must not substitute generated geometry")

    monkeypatch.setattr("evaluation.deterministic.fixtures._write_image", no_geometry)
    case = materialize_scenario_case(scenario, tmp_path / "fixture", use_visual_assets=True)
    assert case.claim.images == ["01-generated.png"]
    copied = case.image_path(case.claim.images[0])
    assert copied.read_bytes() == source.read_bytes()
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == asset["sha256"]
    assert case.claim.declared_damage == ["FRONT_BUMPER"]
    provenance = json.loads((case.root_dir / "fixture_provenance.json").read_text(encoding="utf-8"))
    assert provenance["visual_assets"] == [{**asset, "fixture_filename": "01-generated.png"}]
    assert "copied unchanged" in provenance["vision_semantics"]
    assert "no third-party imagery" not in provenance["provenance"]


def test_real_assets_with_identical_basenames_remain_distinct(visual_source, tmp_path):
    source, asset, scenario = visual_source
    other = tmp_path / "other" / source.name
    other.parent.mkdir()
    with Image.new("RGB", (240, 160), "navy") as image:
        image.save(other)
    second = {**asset, "path": str(other), "sha256": hashlib.sha256(other.read_bytes()).hexdigest()}
    scenario = scenario.model_copy(update={"external_assets": [asset, second]})
    case = materialize_scenario_case(scenario, tmp_path / "fixture", use_visual_assets=True)
    assert case.claim.images == ["01-generated.png", "02-generated.png"]
    assert case.image_path("01-generated.png").read_bytes() == source.read_bytes()
    assert case.image_path("02-generated.png").read_bytes() == other.read_bytes()


def test_deterministic_materialization_keeps_geometry_and_ignores_source(visual_source, tmp_path):
    source, _, scenario = visual_source
    source.unlink()
    case = materialize_scenario_case(scenario, tmp_path / "fixture")
    assert case.claim.images == ["synthetic_geometry.png"]
    with Image.open(case.image_path("synthetic_geometry.png")) as image:
        assert image.size == (128, 128)
    provenance = json.loads((case.root_dir / "fixture_provenance.json").read_text(encoding="utf-8"))
    assert provenance["visual_assets"] == []
    assert "fake detections" in provenance["vision_semantics"]


def test_real_materialization_refuses_missing_assets_without_geometry(visual_source, tmp_path):
    _, _, scenario = visual_source
    scenario = scenario.model_copy(update={"external_assets": []})
    with pytest.raises(ValueError, match="No annotated visual reference images configured"):
        materialize_scenario_case(scenario, tmp_path / "fixture", use_visual_assets=True)
    assert not (tmp_path / "fixture/images/synthetic_geometry.png").exists()


def test_real_materialization_refuses_changed_image_bytes(visual_source, tmp_path):
    source, _, scenario = visual_source
    with Image.new("RGB", (320, 180), "red") as image:
        image.save(source)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        materialize_scenario_case(scenario, tmp_path / "fixture", use_visual_assets=True)
    assert not (tmp_path / "fixture/images").exists()


def test_load_case_preserves_image_origin_in_observation(visual_source, tmp_path):
    _, asset, scenario = visual_source

    class ClaimRecorder(ClaimSteps, ComponentRecorder):
        def seed_demo_claim(self, run):
            pass

    recorder = ClaimRecorder()
    recorder.config = SimpleNamespace(fixture_dir=tmp_path / "fixtures")
    run = RealRun(scenario=scenario)
    recorder.load_case(run)
    assert run.observed.components["claim"].status == "PASS"
    assert run.observed.provenance["visual_assets"] == [{**asset, "fixture_filename": "01-generated.png"}]
    assert run.observed.provenance["controlled_synthetic_claim"] is True


def test_scenario_nine_materializes_approved_generated_asset_unchanged(tmp_path):
    scenario = load_scenarios(scenario_ids=["visual_additional_damage"])[0]
    asset = scenario.external_assets[0]
    source = (ROOT / asset["path"]).resolve()
    case = materialize_scenario_case(scenario, tmp_path / "fixture", use_visual_assets=True)
    filename = f"01-{source.name}"
    assert case.claim.images == [filename]
    copied = case.image_path(filename)
    assert copied.read_bytes() == source.read_bytes()
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == asset["sha256"]
    provenance = json.loads((case.root_dir / "fixture_provenance.json").read_text(encoding="utf-8"))
    assert provenance["visual_assets"] == [{**asset, "path": str(source), "fixture_filename": filename}]
    assert provenance["visual_assets"][0]["origin"] == "ai_generated"
    assert set(provenance["visual_assets"][0]["annotation"]["damaged_parts"]) == {"HOOD", "FRONT_BUMPER"}
