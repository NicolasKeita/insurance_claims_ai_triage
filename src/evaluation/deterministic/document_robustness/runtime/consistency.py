"""Condition downstream evaluation on all three documents matching ground truth."""

from claims.consistency import compare_claim_documents
from claims.document_models import AccidentReportExtraction, ClaimFormExtraction, GarageQuoteExtraction

MODELS = {
    "claim_form": ClaimFormExtraction,
    "accident_report": AccidentReportExtraction,
    "garage_quote": GarageQuoteExtraction,
}


def consistency_results(scenarios, records):
    groups = {}
    for record in records:
        groups.setdefault((record["scenario_id"], record["variant"]), {})[record["document_type"]] = record
    results = []
    expected = {scenario.scenario_id: scenario.expected_consistency for scenario in scenarios}
    for (scenario, variant), documents in sorted(groups.items()):
        evaluable = len(documents) == 3 and all(
            item["structured_metrics"]["exact_match_count"] == item["structured_metrics"]["field_count"]
            and item["structured_metrics"]["parse_success"]
            for item in documents.values()
        )
        actual = None
        if evaluable:
            models = {kind: MODELS[kind].model_validate(item["structured"]) for kind, item in documents.items()}
            actual = compare_claim_documents(
                models["claim_form"], models["accident_report"], models["garage_quote"]
            ).model_dump(mode="json")
        results.append(
            {
                "scenario_id": scenario,
                "variant": variant,
                "evaluable": evaluable,
                "expected": expected[scenario],
                "actual": actual,
                "preserved": actual == expected[scenario] if evaluable else None,
                "reason": None if evaluable else "Three correctly extracted documents are required",
            }
        )
    return results
