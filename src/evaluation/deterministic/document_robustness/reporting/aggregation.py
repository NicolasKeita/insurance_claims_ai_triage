"""Micro-averaged field/text metrics with explicit coverage denominators."""

from statistics import mean

from evaluation.deterministic.document_robustness.metrics import error_rate


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def aggregate(records):
    evaluated = [row for row in records if row["status"] != "not_executed"]
    fields = [row["structured_metrics"] for row in evaluated]

    def total(key):
        return sum(item[key] for item in fields)

    parsed = sum(item["parse_success"] for item in fields)
    metrics = {
        "evaluated_count": len(evaluated),
        "not_executed_count": len(records) - len(evaluated),
        "parse_failure_count": len(evaluated) - parsed,
        "document_parse_success_rate": ratio(parsed, len(evaluated)),
        "field_exact_match_rate": ratio(total("exact_match_count"), total("field_count")),
        "required_field_recall": ratio(total("required_match_count"), total("required_field_count")),
        "required_field_presence_rate": ratio(total("required_present_count"), total("required_field_count")),
        "unexpected_value_count": total("unexpected_value_count"),
        "critical_field_accuracy": ratio(total("critical_match_count"), total("critical_field_count")),
        "field_count": total("field_count"),
        "critical_field_count": total("critical_field_count"),
    }
    metrics.update(_text_aggregate(evaluated))
    metrics.update(
        {
            f"mean_{key}": mean(row[key] for row in evaluated if row[key] is not None)
            if any(row[key] is not None for row in evaluated)
            else None
            for key in (
                "native_duration_seconds",
                "ocr_duration_seconds",
                "structured_duration_seconds",
                "duration_seconds",
            )
        }
    )
    metrics["line_items"] = _line_items(fields)
    return metrics


def _text_aggregate(records):
    def rates(rows):
        values = [row["text_metrics"] for row in rows if row["text_metrics"] is not None]

        def total(key):
            return sum(item[key] for item in values)

        return {
            "cer": error_rate(total("character_errors"), total("reference_characters")) if values else None,
            "wer": error_rate(total("word_errors"), total("reference_words")) if values else None,
            "text_evaluated_count": len(values),
        }

    return {
        **rates(records),
        **{
            f"ocr_{key}": value
            for key, value in rates([row for row in records if row["extraction_method"] == "OCR"]).items()
        },
        **{
            f"hybrid_{key}": value
            for key, value in rates([row for row in records if row["extraction_method"] == "HYBRID"]).items()
        },
    }


def _line_items(fields):
    values = [item["line_items"] for item in fields if item["line_items"] is not None]
    keys = ("expected_count", "extracted_count", "matching_items", "missing_items", "unexpected_items")
    return {key: sum(value[key] for value in values) for key in keys}


def critical_fields(records):
    groups = {}
    for row in records:
        if row["status"] == "not_executed":
            continue
        for field in row["structured_metrics"]["fields"]:
            if field["critical"]:
                name = "damages" if field["field"] in ("declared_damage", "reported_damage") else field["field"]
                groups.setdefault(name, []).append(field)
    return {
        name: {
            "matched": sum(field["matched"] for field in fields),
            "expected": len(fields),
            "accuracy": ratio(sum(field["matched"] for field in fields), len(fields)),
        }
        for name, fields in sorted(groups.items())
    }
