"""Resolve inference thresholds without running scenarios or loading models."""

import pytest

from evaluation import E2E_BENCHMARK_VERSION
from evaluation.preflight.config import RealEvaluationConfig
from evaluation.preflight.console import print_preflight_row

THRESHOLDS = (
    ("damage_confidence_threshold", "DAMAGE_CONFIDENCE_THRESHOLD"),
    ("part_confidence_threshold", "PART_CONFIDENCE_THRESHOLD"),
)


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    for _, variable in THRESHOLDS:
        monkeypatch.delenv(variable, raising=False)
    path = tmp_path / "evaluation.toml"
    path.write_text(
        f'benchmark_version = "{E2E_BENCHMARK_VERSION}"\n'
        "[models]\ndamage_confidence_threshold = 0.35\npart_confidence_threshold = 0.45\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("field,_", THRESHOLDS)
@pytest.mark.parametrize("value", [0, 0.25, 0.5, 1])
def test_thresholds_accept_finite_probabilities(field, _, value):
    assert getattr(RealEvaluationConfig(**{field: value}), field) == value


@pytest.mark.parametrize("field,_", THRESHOLDS)
@pytest.mark.parametrize("value", [-0.1, 1.1, True, "0.5", None, float("nan"), float("inf")])
def test_thresholds_reject_invalid_values_before_model_loading(field, _, value):
    with pytest.raises(ValueError, match=field):
        RealEvaluationConfig(**{field: value})


def test_thresholds_retain_existing_defaults():
    config = RealEvaluationConfig()
    assert config.damage_confidence_threshold == config.part_confidence_threshold == 0.25


def test_toml_thresholds_preserve_independent_damage_and_parts_settings(settings_file):
    config = RealEvaluationConfig.from_toml(settings_file)
    assert config.damage_confidence_threshold == 0.35
    assert config.part_confidence_threshold == 0.45


@pytest.mark.parametrize("field,variable", THRESHOLDS)
def test_environment_overrides_toml_and_explicit_options_override_environment(
    settings_file, monkeypatch, field, variable
):
    monkeypatch.setenv(variable, "0.55")
    assert getattr(RealEvaluationConfig.from_toml(settings_file), field) == 0.55
    assert getattr(RealEvaluationConfig.from_toml(settings_file, **{field: None}), field) == 0.55
    explicit = RealEvaluationConfig.from_toml(settings_file, **{field: 0.65})
    assert getattr(explicit, field) == 0.65


@pytest.mark.parametrize("field,variable", THRESHOLDS)
@pytest.mark.parametrize("value", ["invalid", "nan", "inf", "-0.1", "1.1"])
def test_invalid_environment_thresholds_fail_configuration(settings_file, monkeypatch, field, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError):
        RealEvaluationConfig.from_toml(settings_file)
    assert getattr(RealEvaluationConfig.from_toml(settings_file, **{field: 0.5}), field) == 0.5


def test_cli_parses_explicit_thresholds_without_executing_evaluation(settings_file):
    from scripts.evaluate_end_to_end import build_parser

    args = build_parser().parse_args(["--damage-confidence-threshold", "0.6", "--part-confidence-threshold", "0.4"])
    config = RealEvaluationConfig.from_toml(
        settings_file,
        damage_confidence_threshold=args.damage_confidence_threshold,
        part_confidence_threshold=args.part_confidence_threshold,
    )
    assert config.damage_confidence_threshold == 0.6
    assert config.part_confidence_threshold == 0.4


@pytest.mark.parametrize("component", ["damage_vision", "part_vision"])
def test_vision_progress_reports_effective_inference_settings(capsys, component):
    print_preflight_row(
        {
            "component": component,
            "status": "PASS",
            "confidence_threshold": 0.45,
            "inference_image_size": 640,
        }
    )
    assert "(confidence 0.45; image size 640)" in capsys.readouterr().out
