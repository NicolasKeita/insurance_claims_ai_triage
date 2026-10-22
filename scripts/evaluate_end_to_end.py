"""Run the frozen controlled benchmark, with explicit opt-in real dependencies.

The dedicated script never runs as part of normal pytest discovery. Diagnostics
have a separately versioned manifest and separate output/metrics, outside the
ten golden cases' pass-rate denominator.
"""

import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


class _ProgressPipeline:
    def __init__(self, pipeline, output_dir=None):
        self.pipeline = pipeline
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def run(self, scenario):
        preflight = getattr(self.pipeline, "preflight", None)
        missing = preflight.blocked_scenarios.get(scenario.scenario_id) if preflight is not None else None
        if missing:
            print(f"Skipping {scenario.scenario_id}: " + "; ".join(missing), flush=True)
        else:
            print(f"Running {scenario.scenario_id} ...", flush=True)
        result = self.pipeline.run(scenario)
        # Keep the actual observation even if the process is interrupted before
        # the complete report can be assembled (for example during later LLM calls).
        if self.output_dir is not None:
            _write_json(self.output_dir / "observations" / f"{scenario.scenario_id}.json",
                        result.model_dump(mode="json"))
        statuses = ", ".join(f"{key}={item.status}" for key, item in result.components.items() if key != "total")
        print(f"  {statuses}", flush=True)
        return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("real", "deterministic"), default="real")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation/e2e_v1.toml")
    parser.add_argument("--manifest", type=Path, help="Versioned golden scenario manifest")
    parser.add_argument("--scenario", action="append", help="Golden scenario ID; repeat to select multiple")
    parser.add_argument("--include-vision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-agent", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-rag", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-similar-claims", dest="include_similar", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-diagnostics", action=argparse.BooleanOptionalAction, default=True,
                        help="Run the separate real-only reference/CarDD diagnostic manifest")
    parser.add_argument("--database-url", help="Explicit configured PostgreSQL endpoint; no automatic test endpoint fallback")
    parser.add_argument("--qdrant-url", help="Explicit configured Qdrant endpoint")
    parser.add_argument("--isolated-demo-stack", action="store_true",
                        help="Explicitly seed controlled demos in a new disposable schema and vector collections; requires explicit safe test database and Qdrant URLs")
    parser.add_argument("--llm-model", help="Already installed Ollama model; never downloads/pulls")
    parser.add_argument("--vision-device", help="Use 0 for CUDA or cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/evaluation/e2e_v1")
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dependency-unavailable", choices=("SKIP", "FAIL"), default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--baseline", type=Path, help="Compare stable expectations against a previously successful real run")
    parser.add_argument("--save-baseline", action="store_true", help="Save only after a COMPLETE real benchmark with all PASS")
    return parser


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    from evaluation.scenarios import load_scenarios
    from evaluation.runner import evaluate_scenarios
    from evaluation.reporting import compare_baseline, create_baseline, write_report
    scenarios = load_scenarios(args.manifest, args.scenario)
    preflight, pipeline = None, None
    contexts = ExitStack()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "deterministic":
            if args.preflight_only:
                raise ValueError("--preflight-only applies to --mode real")
            from evaluation.pipeline import ClaimsEvaluationPipeline
            pipeline = ClaimsEvaluationPipeline()
            fail_fast = bool(args.fail_fast)
        else:
            from evaluation.preflight import prepare_preflight, print_preflight_row
            from evaluation.real import RealClaimsEvaluationPipeline
            names = ("include_vision", "include_agent", "include_rag", "include_similar", "fail_fast",
                     "dependency_unavailable", "database_url", "qdrant_url", "llm_model", "vision_device", "isolated_demo_stack")
            print("Real-stack prerequisites (no model loading): component / status / version / reason", flush=True)
            config, preflight = prepare_preflight(
                args.config, scenarios,
                overrides={**{name: getattr(args, name) for name in names}, "fixture_dir": args.output_dir / "fixtures"},
                progress=print_preflight_row, load_models=not args.preflight_only,
            )
            _write_json(args.output_dir / "preflight.json", preflight.rows)
            if args.preflight_only:
                print(f"Preflight saved: {(args.output_dir / 'preflight.json').resolve()}")
                return int(any(row["status"] == "FAIL" for row in preflight.rows))
            runnable = any(item.scenario_id not in preflight.blocked_scenarios for item in scenarios)
            if config.isolated_demo_stack and runnable:
                from evaluation.real import isolated_demo_stack
                _write_json(args.output_dir / "preflight_before_isolated_setup.json", preflight.rows)
                missing = [name for name in ("postgresql", "qdrant", "embeddings") if not preflight.available(name)]
                if missing:
                    reason = "Isolated setup unavailable: " + ", ".join(missing)
                    print(reason + "; independent real components will still be evaluated", flush=True)
                    _write_json(args.output_dir / "isolated_setup.json", {"status": config.dependency_unavailable,
                        "reason": reason, "created_resources": [], "cleaned_up": True})
                    config = replace(config, isolated_demo_stack=False)
                else:
                    print("Preparing explicitly requested isolated demo schema and vector collections ...", flush=True)
                    contexts.enter_context(isolated_demo_stack(config, preflight, args.output_dir))
                _write_json(args.output_dir / "preflight.json", preflight.rows)
            pipeline = RealClaimsEvaluationPipeline(config, preflight)
            fail_fast = config.fail_fast

        report = evaluate_scenarios(scenarios, _ProgressPipeline(pipeline, args.output_dir), mode=args.mode,
            preflight=preflight.rows if preflight is not None else [], fail_fast=fail_fast)
        write_report(report, args.output_dir)
        counts = report.metrics["scenario_counts"]
        print(f"Golden benchmark: PASS {counts['PASS']} / FAIL {counts['FAIL']} / SKIP {counts['SKIP']}")
        print(f"Full-stack status: {report.full_stack_status}")
        print(f"Artifacts: {args.output_dir.resolve()}")
        failed = counts["FAIL"] > 0

        if args.mode == "real" and args.include_diagnostics and runnable:
            from evaluation.real import real_diagnostic_scenarios
            diagnostics = evaluate_scenarios(real_diagnostic_scenarios(), _ProgressPipeline(pipeline, args.output_dir / "diagnostics"), mode="real",
                preflight=preflight.rows, fail_fast=False)
            diagnostics.limitations.append("DIAGNOSTICS ONLY: excluded from the golden benchmark pass-rate denominator; no image detection ground truth")
            write_report(diagnostics, args.output_dir / "diagnostics")
            print(f"Separate diagnostic artifacts: {(args.output_dir / 'diagnostics').resolve()}")
            failed |= any(result.status == "FAIL" for result in diagnostics.results)

        if args.mode == "real" and args.include_diagnostics and not runnable:
            print("Diagnostics skipped: no selected golden scenario passed prerequisites", flush=True)

        if args.save_baseline:
            path = args.output_dir / "baseline_summary.json"
            saved = create_baseline(report, path)
            _write_json(args.output_dir / "baseline_status.json", {
                "created": saved, "baseline_path": str(path.resolve()) if saved else None,
                "reason": None if saved else "No complete successful real-stack benchmark; partial results cannot establish a regression baseline",
            })
            print(f"Baseline {'saved: ' + str(path.resolve()) if saved else 'withheld: complete successful real stack required'}")
        if args.baseline:
            regressions = compare_baseline(report, args.baseline)
            _write_json(args.output_dir / "baseline_comparison.json", {"regressions": regressions})
            print(f"Baseline regressions: {len(regressions)}")
            failed |= bool(regressions)
        return int(failed)
    finally:
        contexts.close()
        if preflight is not None:
            preflight.close()


if __name__ == "__main__":
    raise SystemExit(main())
