"""Visual file validation and scenario-scoped eligibility without inference or E2E."""

import hashlib

from PIL import Image
import pytest

from evaluation.preflight import run_preflight
from evaluation.preflight.console import print_preflight_row
from evaluation.scenarios import load_scenarios
from evaluation.visual_assets import visual_assets
from tests.infrastructure.prerequisites._evaluation import gate as gate


@pytest.fixture
def asset(tmp_path):
    path = tmp_path / "generated.png"
    with Image.new("RGB", (16, 16), "silver") as image:
        image.save(path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "origin": "ai_generated",
        "provenance": "Unit test image, no real photograph.",
        "annotation": {"damaged_parts": ["HOOD", "FRONT_BUMPER"]},
    }


def with_asset(scenario, asset):
    return scenario.model_copy(update={
        "scenario_id": "visual_ready", "vision_required": True,
        "infrastructure_requirements": ["visual_golden_asset"], "external_assets": [asset],
    })


def test_validated_images_retain_origin_hash_and_reference_labels(gate, asset):
    _, scenario = gate
    assert visual_assets(with_asset(scenario, asset)) == [asset]


def test_explicit_negative_reference_annotation_is_distinct_from_missing_labels(gate, asset):
    asset["annotation"] = {"damaged_parts": []}
    assert visual_assets(with_asset(gate[1], asset))[0]["annotation"]["damaged_parts"] == []


@pytest.mark.parametrize("changes,reason", [
    ({"path": ""}, "path"),
    ({"path": "absent-reference.png"}, "does not exist"),
    ({"sha256": "0" * 64}, "SHA-256 mismatch"),
    ({"origin": "unknown"}, "origin"),
    ({"provenance": " "}, "provenance"),
    ({"annotation": {}}, "damaged_parts"),
    ({"annotation": {"damaged_parts": ["invented_part"]}}, "DamageType"),
])
def test_missing_or_unverified_references_are_rejected(gate, asset, changes, reason):
    _, scenario = gate
    with pytest.raises(ValueError, match=reason):
        visual_assets(with_asset(scenario, {**asset, **changes}))


def test_corrupt_images_cannot_pass_with_a_matching_hash(gate, asset):
    from pathlib import Path

    path = Path(asset["path"])
    path.write_bytes(b"Not a decoded image")
    asset["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="safely decoded"):
        visual_assets(with_asset(gate[1], asset))


def test_passing_reference_does_not_unblock_another_visual_scenario(gate, asset):
    config, scenario = gate
    ready = with_asset(scenario, asset)
    missing = ready.model_copy(update={"scenario_id": "visual_missing", "external_assets": []})
    report = run_preflight(config, scenarios=[ready, missing], load_models=False)
    assert ready.scenario_id not in report.blocked_scenarios
    assert missing.scenario_id in report.blocked_scenarios
    assert report.available("visual_golden_asset:visual_ready")
    assert not report.available("visual_golden_asset:visual_missing")
    assert "visual_golden_asset:visual_missing" in next(
        row["missing_components"] for row in report.rows if row["component"] == "prerequisites:visual_missing"
    )


def test_approved_scenario_nine_reference_is_available_and_explicitly_generated(capsys):
    scenario = load_scenarios(scenario_ids=["visual_additional_damage"])[0]
    assets = visual_assets(scenario)
    assert len(assets) == 1
    assert assets[0]["origin"] == "ai_generated"
    assert set(assets[0]["annotation"]["damaged_parts"]) == {"HOOD", "FRONT_BUMPER"}
    assert "not a CarDD photograph" in assets[0]["provenance"]
    print_preflight_row({"component": f"visual_golden_asset:{scenario.scenario_id}", "status": "PASS", "assets": assets})
    assert "Annotated visual reference image (AI-generated)" in capsys.readouterr().out
