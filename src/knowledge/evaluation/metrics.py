"""Recall@K and MRR over exact source_id + section ground truth."""


def section_metrics(results, expected_source_id, expected_sections):
    truth = {(expected_source_id, s) for s in expected_sections}
    if not truth:
        raise ValueError("Recall/MRR require a nonempty answerable ground truth")
    values = {}
    for k in (1, 3, 5):
        found = {(r.source_id, r.section) for r in results[:k]}
        values[f"recall_at_{k}"] = len(found & truth) / len(truth)
    values["mrr"] = next((1 / rank for rank, r in enumerate(results[:5], 1) if (r.source_id, r.section) in truth), 0.0)
    return values
