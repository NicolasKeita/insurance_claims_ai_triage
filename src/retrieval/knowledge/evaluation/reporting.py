"""JSON + CSV artifacts for a retrieval benchmark run."""

import csv
import json
from pathlib import Path

CSV_FIELDS = ["id", "query", "answerable", "rank", "chunk_id", "source_id", "section", "similarity_score"]


def write_evaluation(report: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "retrieval_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in report["cases"]:
            for rank, result in enumerate(row["results"], 1):
                writer.writerow(_csv_row(row, rank, result))


def _csv_row(row, rank, result):
    return {
        "id": row["case"]["id"],
        "query": row["case"]["query"],
        "answerable": row["case"]["answerable"],
        "rank": rank,
        **{k: result[k] for k in ("chunk_id", "source_id", "section", "similarity_score")},
    }
