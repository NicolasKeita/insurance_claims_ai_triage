"""Read-only dependency probes for the explicit, opt-in real evaluation.

Heavy adapters are loaded once and retained for the run. No downloaded models,
service startup, writes to the database/vector stores, or fake replacements.
"""

from dataclasses import replace

from evaluation.preflight.config import DEFAULT_CONFIG, ROOT, RealEvaluationConfig
from evaluation.preflight.report import PreflightReport, checkpoint_version, print_preflight_row
from evaluation.preflight.session import PreflightSession

__all__ = [
    "DEFAULT_CONFIG",
    "ROOT",
    "PreflightReport",
    "PreflightSession",
    "RealEvaluationConfig",
    "checkpoint_version",
    "print_preflight_row",
    "prepare_preflight",
    "run_preflight",
]


def prepare_preflight(path, scenarios, *, overrides=None, progress=None, load_models=True):
    """Treat a missing or invalid configuration as an explicit prerequisite skip."""
    overrides = overrides or {}
    fixture_dir = overrides.get("fixture_dir")
    settings = {name: value for name, value in overrides.items() if name != "fixture_dir"}
    try:
        config = RealEvaluationConfig.from_toml(path, **settings)
    except (OSError, ValueError, TypeError) as error:
        config = RealEvaluationConfig(dependency_unavailable=overrides.get("dependency_unavailable") or "SKIP")
        reason = f"configuration: {type(error).__name__}: {error}"
        row = {"component": "configuration", "status": config.dependency_unavailable, "version": None, "reason": reason}
        report = PreflightReport(rows=[row], blocked_scenarios={item.scenario_id: [reason] for item in scenarios})
        if progress is not None:
            progress(row)
        return config, report
    if fixture_dir is not None:
        config = replace(config, fixture_dir=fixture_dir)
    return config, run_preflight(config, progress=progress, scenarios=scenarios, load_models=load_models)


def run_preflight(config: RealEvaluationConfig, *, progress=None, scenarios=None, load_models=True) -> PreflightReport:
    """Check selected scenarios before loading models; retain service clients."""
    session = PreflightSession(config, progress)
    try:
        if scenarios is None:
            return session.run()
        from evaluation.preflight.prerequisites import PrerequisiteChecks

        PrerequisiteChecks(session).check(scenarios)
        runnable = [item for item in scenarios if item.scenario_id not in session.report.blocked_scenarios]
        if runnable and load_models:
            session.probe_embeddings()
            session.probe_mlflow()
            session.probe_models()
            session.probe_detectors()
            from evaluation.preflight.report import record_eligibility, requirements

            for scenario in runnable:
                record_eligibility(session, scenario, requirements(config, scenario), models_loaded=True)
        return session.report
    except Exception:
        session.report.close()
        raise
