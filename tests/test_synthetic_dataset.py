import numpy as np
import pandas as pd

from claims.enums import TriageWorkflow
from ml.synthetic_dataset import (
    generate_synthetic_triage_dataset,
)

def test_dataset_size():
    dataset = (
        generate_synthetic_triage_dataset(
            sample_count=500,
            seed=42,
        )
    )

    assert len(dataset) == 500

def test_dataset_is_reproducible():
    first = (
        generate_synthetic_triage_dataset(
            sample_count=100,
            seed=42,
        )
    )

    second = (
        generate_synthetic_triage_dataset(
            sample_count=100,
            seed=42,
        )
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )

def test_document_completeness_is_consistent():
    dataset = (
        generate_synthetic_triage_dataset(
            sample_count=500,
            seed=42,
        )
    )

    expected = (
        dataset["document_count"]
        - dataset[
            "missing_document_count"
        ]
    ) / dataset["document_count"]

    assert np.allclose(
        dataset[
            "document_completeness_ratio"
        ],
        expected,
    )

def test_all_workflows_are_generated():
    dataset = (
        generate_synthetic_triage_dataset(
            sample_count=5000,
            seed=42,
        )
    )

    assert set(
        dataset["workflow"]
    ) == {
        workflow.value
        for workflow in TriageWorkflow
    }

def test_no_workflow_is_too_rare():
    dataset = (
        generate_synthetic_triage_dataset(
            sample_count=5000,
            seed=42,
        )
    )

    frequencies = (
        dataset["workflow"]
        .value_counts(
            normalize=True
        )
    )

    assert (
        frequencies.min()
        > 0.05
    )

def test_repair_amount_does_not_define_workflow_alone():
    dataset = (
        generate_synthetic_triage_dataset(
            sample_count=5000,
            seed=42,
        )
    )

    middle_band = dataset[
        dataset["repair_amount"].between(
            2000,
            4000,
        )
    ]

    assert (
        middle_band["workflow"].nunique()
        >= 3
    )