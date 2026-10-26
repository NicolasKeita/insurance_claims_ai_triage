"""Run the real golden benchmark serially, releasing each process before resting."""

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

VALUE_OPTIONS = (
    "config",
    "manifest",
    "llm_model",
    "database_url",
    "qdrant_url",
    "vision_device",
    "damage_confidence_threshold",
    "part_confidence_threshold",
    "vision_image_size",
    "vision_max_temperature_c",
)
SWITCH_OPTIONS = ("include_agent", "include_rag", "include_similar_claims")
STATUSES = ("PASS", "FAIL", "SKIP", "STOPPED", "NOT_RUN")


def _cooldown(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("cooldown must be finite and non-negative")
    return seconds


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("real",), default="real")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation/e2e_v2_damage960_val640.toml")
    parser.add_argument("--manifest", type=Path, help="Versioned golden scenario manifest")
    parser.add_argument("--scenario", action="append", help="Optional subset, always run in manifest order")
    parser.add_argument("--llm-model", help="Already installed Ollama model; never downloads/pulls")
    parser.add_argument("--database-url")
    parser.add_argument("--qdrant-url")
    parser.add_argument("--isolated-demo-stack", action="store_true")
    parser.add_argument(
        "--include-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load vision only for scenarios requiring it (default: enabled)",
    )
    for name in SWITCH_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--no-include-diagnostics",
        action="store_true",
        help="Accepted for compatibility; this suite always excludes diagnostics",
    )
    parser.add_argument("--vision-device", help="Use 0 for CUDA or cpu")
    parser.add_argument("--damage-confidence-threshold", type=float)
    parser.add_argument("--part-confidence-threshold", type=float)
    parser.add_argument("--vision-image-size", type=int)
    parser.add_argument(
        "--vision-max-temperature-c", type=float, help="Forward the existing thermal cutoff (profile default: 80 C)"
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=_cooldown,
        default=30.0,
        help="Rest after process exit and before the next scenario (default: 30)",
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop on any incomplete scenario; safety stops always abort (default: enabled)",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/evaluation/e2e_v2/serial")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the plan without starting processes, contacting services or loading models",
    )
    return parser


def _command(args, scenario, output_dir):
    command = [sys.executable, "-u", str(ROOT / "scripts/evaluate_end_to_end.py"), "--mode", "real"]
    for name in VALUE_OPTIONS:
        value = getattr(args, name)
        if value is not None:
            command.extend(("--" + name.replace("_", "-"), str(value)))
    for name in SWITCH_OPTIONS:
        value = getattr(args, name)
        if value is not None:
            command.append(("--" if value else "--no-") + name.replace("_", "-"))
    if args.isolated_demo_stack:
        command.append("--isolated-demo-stack")
    command.extend(
        (
            "--scenario",
            scenario.scenario_id,
            "--output-dir",
            str(output_dir),
            "--include-vision" if args.include_vision and scenario.vision_required else "--no-include-vision",
            "--no-include-diagnostics",
        )
    )
    return command


def _result(scenario_id, output_dir, returncode):
    result = {
        "scenario_id": scenario_id,
        "status": "FAIL",
        "returncode": returncode,
        "output_dir": str(output_dir),
        "reason": None,
    }
    safety_stop = output_dir / "safety_stop.json"
    if safety_stop.exists():
        reason = "Vision safety stop"
        try:
            reason += ": " + str(json.loads(safety_stop.read_text(encoding="utf-8"))["reason"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        result.update(status="STOPPED", reason=reason + "; inspect " + str(safety_stop))
        return result
    try:
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        rows = summary["scenarios"]
        if (
            summary["mode"] != "real"
            or len(rows) != 1
            or rows[0]["scenario_id"] != scenario_id
            or rows[0]["status"] not in {"PASS", "FAIL", "SKIP"}
        ):
            raise ValueError("Expected exactly the selected real scenario in the report")
        status = rows[0]["status"]
        complete = summary["full_stack_status"] == "COMPLETE"
        if returncode != 0:
            result["reason"] = f"Evaluator exited with code {returncode}"
        elif status == "PASS" and not complete:
            result["reason"] = "Scenario assertions passed but the full stack is PARTIAL"
        else:
            result["status"] = status
            if status != "PASS":
                result["reason"] = f"Scenario reported {status}; inspect its report"
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result["reason"] = f"Missing or invalid scenario report: {error}"
    return result


def _save_summary(path, summary):
    results = summary["results"]
    summary["counts"] = {status: sum(row["status"] == status for row in results) for status in STATUSES}
    summary["full_stack_status"] = "COMPLETE" if all(row["status"] == "PASS" for row in results) else "PARTIAL"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_child(command):
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        return process.wait()
    except KeyboardInterrupt:
        # Console Ctrl+C reaches the child too. Give its finally blocks time to
        # remove the isolated database schema/collections and release models.
        print("Interrupted; waiting up to 30s for evaluator cleanup ...", flush=True)
        try:
            process.wait(timeout=30)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            process.terminate()
            try:
                process.wait(timeout=10)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                process.kill()
                process.wait()
        raise


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    from evaluation.scenarios import load_scenarios

    scenarios = load_scenarios(args.manifest, args.scenario)
    print(
        f"Serial real benchmark: {len(scenarios)} scenarios; rest {args.cooldown_seconds:g}s between processes.",
        flush=True,
    )
    for index, scenario in enumerate(scenarios, 1):
        vision = "enabled" if args.include_vision and scenario.vision_required else "disabled"
        print(f"  {index}/{len(scenarios)} {scenario.scenario_id} (vision {vision})", flush=True)
    if args.dry_run:
        print("Dry run: no scenario executed, no models loaded, no services contacted.", flush=True)
        return 0

    stamp = datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir.resolve() / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "suite_summary.json"
    summary = {
        "mode": "real",
        "cooldown_seconds": args.cooldown_seconds,
        "stopped_reason": None,
        "results": [
            {
                "scenario_id": scenario.scenario_id,
                "status": "NOT_RUN",
                "returncode": None,
                "output_dir": str(output_dir / scenario.scenario_id),
                "reason": None,
            }
            for scenario in scenarios
        ],
    }
    _save_summary(summary_path, summary)
    print(f"Suite report: {summary_path}", flush=True)
    interrupted = False
    active_index = None
    try:
        for index, scenario in enumerate(scenarios):
            if index and args.cooldown_seconds:
                print(f"Resting {args.cooldown_seconds:g}s before {scenario.scenario_id} ...", flush=True)
                time.sleep(args.cooldown_seconds)
            scenario_dir = output_dir / scenario.scenario_id
            active_index = index
            print(f"[{index + 1}/{len(scenarios)}] Starting {scenario.scenario_id} ...", flush=True)
            try:
                returncode = _run_child(_command(args, scenario, scenario_dir))
                result = _result(scenario.scenario_id, scenario_dir, returncode)
            except OSError as error:
                result = dict(summary["results"][index], status="FAIL", reason=f"Cannot start evaluator: {error}")
            summary["results"][index] = result
            active_index = None
            _save_summary(summary_path, summary)
            print(
                f"  {scenario.scenario_id}: {result['status']}"
                + (f" ({result['reason']})" if result["reason"] else ""),
                flush=True,
            )
            if result["status"] == "STOPPED" or (args.fail_fast and result["status"] != "PASS"):
                summary["stopped_reason"] = result["reason"]
                break
    except KeyboardInterrupt:
        interrupted = True
        summary["stopped_reason"] = "Interrupted by user; no further scenario started"
        if active_index is not None:
            summary["results"][active_index].update(status="STOPPED", reason="Interrupted by user")
        print(summary["stopped_reason"], flush=True)
    finally:
        _save_summary(summary_path, summary)
    counts = summary["counts"]
    print("Serial benchmark: " + " / ".join(f"{status} {counts[status]}" for status in STATUSES), flush=True)
    print(f"Full-stack status: {summary['full_stack_status']}; report: {summary_path}", flush=True)
    return 130 if interrupted else int(summary["full_stack_status"] != "COMPLETE")


if __name__ == "__main__":
    raise SystemExit(main())
