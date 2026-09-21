"""Tests for the independent compositional promotion benchmark."""

from __future__ import annotations

import csv

from optcon.benchmarks.compositional_protocol import (
    DEFAULT_CASE_OUTPUT_PATH,
    DEFAULT_SUMMARY_OUTPUT_PATH,
    MANIFEST_HASH,
    PROTOCOL_MANIFEST,
    protocol_summary,
    run_protocol_benchmark,
    write_protocol_results,
    write_protocol_summary,
)


def test_protocol_manifest_has_24_frozen_cases_across_four_families():
    ids = [case.case_id for case in PROTOCOL_MANIFEST]

    assert len(ids) == 24
    assert len(ids) == len(set(ids))
    assert sum(not case.is_fault for case in PROTOCOL_MANIFEST) == 12
    assert sum(case.is_fault for case in PROTOCOL_MANIFEST) == 12
    assert {case.family for case in PROTOCOL_MANIFEST} == {
        "mie",
        "thinfilm",
        "beam",
        "decision",
    }
    assert len(MANIFEST_HASH) == 64


def test_protocol_benchmark_accepts_controls_and_blocks_all_faults():
    results = run_protocol_benchmark()
    summary = protocol_summary(results)

    assert len(results) == 24
    assert summary.detected_faults == 12
    assert summary.accepted_controls == 12
    assert summary.faulty_decision_ready == 0
    assert summary.fault_detection_rate == 1.0
    assert summary.control_acceptance_rate == 1.0
    assert summary.incorrect_promotion_rate == 0.0
    assert all(result.expected_outcome_met for result in results)
    assert all(result.manifest_hash == MANIFEST_HASH for result in results)


def test_protocol_benchmark_writes_auditable_case_and_summary_csv(tmp_path):
    results = run_protocol_benchmark()
    case_output = tmp_path / "cases.csv"
    summary_output = tmp_path / "summary.csv"

    assert write_protocol_results(results, case_output) == case_output
    assert write_protocol_summary(protocol_summary(results), summary_output) == summary_output

    with case_output.open(newline="", encoding="utf-8") as stream:
        case_rows = list(csv.DictReader(stream))
    with summary_output.open(newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))

    assert len(case_rows) == 24
    assert len(summary_rows) == 1
    assert {
        "case_id",
        "family",
        "decision_ready",
        "detected",
        "stable",
        "margin",
        "blocking_reasons",
    } <= set(case_rows[0])
    assert summary_rows[0]["faulty_decision_ready"] == "0"


def test_protocol_default_outputs_are_artifact_paths():
    assert "optcon-artifacts" in DEFAULT_CASE_OUTPUT_PATH.parts
    assert "optcon-artifacts" in DEFAULT_SUMMARY_OUTPUT_PATH.parts
    assert "private-paper" not in DEFAULT_CASE_OUTPUT_PATH.parts
