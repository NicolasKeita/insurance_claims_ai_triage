from pathlib import Path

from ml.tracking import (
    compute_file_sha256,
)

def test_file_sha256_is_reproducible(
    tmp_path: Path,
):
    file_path = (
        tmp_path
        / "dataset.csv"
    )

    file_path.write_text(
        "a,b\n1,2\n",
        encoding="utf-8",
    )

    first = compute_file_sha256(
        file_path
    )

    second = compute_file_sha256(
        file_path
    )

    assert first == second

def test_file_sha256_changes_with_content(
    tmp_path: Path,
):
    file_path = (
        tmp_path
        / "dataset.csv"
    )

    file_path.write_text(
        "a,b\n1,2\n",
        encoding="utf-8",
    )

    first = compute_file_sha256(
        file_path
    )

    file_path.write_text(
        "a,b\n1,3\n",
        encoding="utf-8",
    )

    second = compute_file_sha256(
        file_path
    )

    assert first != second