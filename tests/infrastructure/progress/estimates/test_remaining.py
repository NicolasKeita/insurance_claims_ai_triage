"""Remaining time follows recorded stages without claiming stalled work is finished."""

import pytest

from evaluation.real.progress.estimate import HistoricalEstimate, eta_text


def test_estimate_replaces_finished_stage_with_progress_even_when_it_was_slower():
    estimate = HistoricalEstimate({"total": 100, "document": 20, "rag_generation": 30, "agent": 40})
    assert estimate.snapshot(0)["estimated_remaining_seconds"] == 100
    assert estimate.snapshot(5, "document", 5)["estimated_remaining_seconds"] == 95

    estimate.completed.add("document")
    after_document = estimate.snapshot(35)
    assert after_document == {
        "estimated_remaining_seconds": 80,
        "eta_reason": None,
        "reference_total_seconds": 100,
    }
    assert estimate.snapshot(45, "rag_generation", 10)["estimated_remaining_seconds"] == 70
    assert eta_text(after_document) == "ETA ~80.0s remaining (previous run)"


def test_skipped_stages_remove_their_recorded_time_and_keep_recorded_overhead():
    estimate = HistoricalEstimate({"total": 100, "document": 20, "rag_generation": 30, "agent": 40})
    estimate.completed.update({"document", "rag_generation"})
    assert estimate.snapshot(35)["estimated_remaining_seconds"] == 50
    estimate.completed.add("agent")
    assert estimate.snapshot(70)["estimated_remaining_seconds"] == 10


def test_aggregate_ml_duration_is_not_counted_as_another_remaining_stage():
    estimate = HistoricalEstimate({"total": 100, "ml": 90, "document": 20, "agent": 70})
    estimate.completed.add("document")
    assert estimate.snapshot(25)["estimated_remaining_seconds"] == 80


@pytest.mark.parametrize("stage_elapsed", [30, 45])
def test_stage_at_or_beyond_reference_reports_unknown_until_it_finishes(stage_elapsed):
    estimate = HistoricalEstimate({"total": 100, "rag_generation": 30, "agent": 60})
    stalled = estimate.snapshot(stage_elapsed, "rag_generation", stage_elapsed)
    assert stalled["estimated_remaining_seconds"] is None
    assert stalled["eta_reason"] == "reference_exceeded"
    assert "ETA unknown" in eta_text(stalled)
    assert "~0.0s" not in eta_text(stalled)

    estimate.completed.add("rag_generation")
    assert estimate.snapshot(stage_elapsed)["estimated_remaining_seconds"] == 70


def test_unknown_active_stage_does_not_reuse_other_stages_as_an_estimate():
    estimate = HistoricalEstimate({"total": 100, "agent": 90})
    unrecorded = estimate.snapshot(5, "document", 5)
    assert unrecorded == {
        "estimated_remaining_seconds": None,
        "eta_reason": "unrecorded_stage",
        "reference_total_seconds": 100,
    }
    assert "ETA unknown" in eta_text(unrecorded)


def test_no_matching_history_has_an_explicit_unknown_eta():
    missing = HistoricalEstimate({}).snapshot(5, "agent", 5)
    assert missing == {
        "estimated_remaining_seconds": None,
        "eta_reason": "no_matching_history",
        "reference_total_seconds": None,
    }
    assert "ETA unknown" in eta_text(missing)
    assert "no matching completed real run" in eta_text(missing)


def test_total_only_reference_counts_elapsed_time_and_stops_at_its_limit():
    estimate = HistoricalEstimate({"total": 62.5, "ml": 60})
    assert estimate.snapshot(12.5, "agent", 5)["estimated_remaining_seconds"] == 50
    for elapsed in (62.5, 80):
        exceeded = estimate.snapshot(elapsed, "agent", elapsed)
        assert exceeded["estimated_remaining_seconds"] is None
        assert exceeded["eta_reason"] == "reference_exceeded"
        assert "ETA unknown" in eta_text(exceeded)


def test_overlapping_recorded_stages_do_not_create_negative_overhead():
    estimate = HistoricalEstimate({"total": 40, "document": 20, "agent": 30})
    estimate.completed.add("document")
    assert estimate.snapshot(25)["estimated_remaining_seconds"] == 30
