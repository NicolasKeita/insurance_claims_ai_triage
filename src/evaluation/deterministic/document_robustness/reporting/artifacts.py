"""Human-readable diagnostics, CSV exports and retained OCR text."""

import csv

from evaluation.deterministic.document_robustness.runtime.storage import write_json


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def document_rows(records):
    rows = []
    for record in records:
        fields = record["structured_metrics"]
        rows.append(
            {
                key: record[key]
                for key in (
                    "scenario_id",
                    "document_id",
                    "document_type",
                    "variant",
                    "status",
                    "extraction_method",
                    "native_duration_seconds",
                    "ocr_duration_seconds",
                    "structured_duration_seconds",
                    "duration_seconds",
                )
            }
        )
        rows[-1].update(
            parse_success=fields["parse_success"],
            exact_match_count=fields["exact_match_count"],
            field_count=fields["field_count"],
            **(record["text_metrics"] or {}),
        )
        rows[-1].update({f"line_items_{key}": value for key, value in (fields["line_items"] or {}).items()})
    return rows


def write_debug_artifacts(output, records):
    text_dir, failures_dir = output / "ocr_text", output / "failures"
    text_dir.mkdir(exist_ok=True)
    failures_dir.mkdir(exist_ok=True)
    critical = []
    for record in records:
        text = record["extracted_text"]
        (text_dir / f"{record['case_key']}.txt").write_text(text, encoding="utf-8")
        failures = []
        for field in record["structured_metrics"]["fields"]:
            context = _context(text, field)
            if field["critical"]:
                critical.append(
                    {
                        "document_id": record["document_id"],
                        "document_type": record["document_type"],
                        "variant": record["variant"],
                        **field,
                    }
                )
            if not field["matched"]:
                failures.append({**field, "extracted_text_context": context})
        if failures or record["errors"]:
            write_json(
                failures_dir / f"{record['case_key']}.json",
                {
                    key: record.get(key)
                    for key in (
                        "document_id",
                        "document_type",
                        "variant",
                        "source_variant",
                        "degradation_config",
                        "errors",
                        "status",
                    )
                }
                | {"fields": failures},
            )
    write_csv(output / "critical_fields.csv", critical)


def _context(text, field):
    for value in (field["actual"], field["expected"]):
        if isinstance(value, (str, int, float)) and str(value) in text:
            index = text.index(str(value))
            return text[max(0, index - 120) : index + len(str(value)) + 120]
    return text[:600]


def print_table(summary):
    def formatted(value):
        return f"{value:.4f}" if value is not None else "N/A"

    print("| Variant | CER | WER | Field exact match | Critical fields | Parse success |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for variant, metrics in summary["by_variant"].items():
        values = [
            formatted(metrics[key])
            for key in (
                "cer",
                "wer",
                "field_exact_match_rate",
                "critical_field_accuracy",
                "document_parse_success_rate",
            )
        ]
        print(f"| {variant} | {' | '.join(values)} |")
