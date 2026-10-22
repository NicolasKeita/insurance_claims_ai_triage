"""Concise terminal output; full diagnostic rows stay in the JSON artifacts."""

from pathlib import Path

from evaluation.preflight.console.messages import PARENTS, PURPOSES, explanation, label

MODELS = {"triage", "anomaly", "embeddings", "damage_vision", "part_vision"}


def print_preflight_row(row):
    """Show availability progress; summarize blocking causes once after checking."""
    name = row["component"]
    if row["status"] != "PASS" or name.startswith("prerequisites:"):
        return
    if name in {"retrieval_configuration", "isolated_configuration", "mlflow"}:
        return
    version = f" ({row['version']})" if name == "ollama" and row.get("version") else ""
    print(f"  [OK] {label(name)}{version}", flush=True)
    if name in PURPOSES:
        print(f"    Required for: {PURPOSES[name]}", flush=True)


def blocking_groups(report, scenario_id):
    """Collapse dependent failures under the missing upstream component."""
    row = next((item for item in report.rows if item["component"] == f"prerequisites:{scenario_id}"), {})
    missing = row.get("missing_components")
    if missing is None:
        missing = [item.partition(": ")[0] for item in report.blocked_scenarios.get(scenario_id, [])]

    def root(name):
        parents = PARENTS.get(name.split(":", 1)[0], ())
        parent = next((item for item in parents if item in missing), None)
        return root(parent) if parent else name

    groups = {}
    for name in missing:
        groups.setdefault(root(name), [])
        if root(name) != name:
            groups[root(name)].append(name)
    return groups


def print_blocked_scenario(report, scenario_id):
    groups = blocking_groups(report, scenario_id)
    row = next((item for item in report.rows if item["component"] == f"prerequisites:{scenario_id}"), {})
    status = row.get(
        "status", next((item["status"] for item in report.rows if item["component"] == "configuration"), "SKIP")
    )
    print(f"\n[{status}] {scenario_id}: test not run.", flush=True)
    print(f"  Blocking components: {len(groups)}", flush=True)
    for name, dependents in groups.items():
        reason = next((item.get("reason", "") for item in report.rows if item["component"] == name), "Not checked")
        cause, action = explanation(name, reason)
        print(f"  - {label(name)}: {cause}.", flush=True)
        if name in PURPOSES:
            print(f"    Required for: {PURPOSES[name]}", flush=True)
        print(f"    Action: {action}", flush=True)
        if dependents:
            print("    Not checked: " + ", ".join(label(item) for item in dependents) + ".", flush=True)


def print_preflight_summary(report, scenarios, output_dir):
    for scenario in scenarios:
        if scenario.scenario_id in report.blocked_scenarios:
            print_blocked_scenario(report, scenario.scenario_id)
        else:
            print(f"\n[READY] {scenario.scenario_id}: prerequisites available.", flush=True)
    all_blocked = all(item.scenario_id in report.blocked_scenarios for item in scenarios)
    if all_blocked and not any(row["component"] in MODELS for row in report.rows):
        print("\nNo models loaded; no scenarios executed.", flush=True)
    print(f"Technical details: {(Path(output_dir) / 'preflight.json').resolve()}", flush=True)
