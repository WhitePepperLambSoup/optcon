"""Tests for the frozen held-out mutation campaign."""

from __future__ import annotations

import csv

from optcon.benchmarks.heldout_mutations import (
    DEFAULT_OUTPUT_PATH,
    HELDOUT_MUTATION_MANIFEST,
    run_heldout_mutations,
    summarize_heldout_mutations,
    write_heldout_results,
)


def test_heldout_manifest_covers_three_optical_families_without_reusing_core_ids():
    ids = [definition.mutation_id for definition in HELDOUT_MUTATION_MANIFEST]

    assert len(ids) == 12
    assert len(ids) == len(set(ids))
    assert {definition.family for definition in HELDOUT_MUTATION_MANIFEST} == {
        "mie",
        "thinfilm",
        "beam",
    }
    assert not set(ids) & {
        "fault-mie-medium-convention",
        "fault-omitted-quadrature-metric",
        "fault-unnormalized-fft-declared-unitary",
    }


def test_heldout_campaign_detects_all_frozen_mutations_and_accepts_controls():
    results = run_heldout_mutations()
    summary = summarize_heldout_mutations(results)

    assert summary["faults_total"] == 9
    assert summary["controls_total"] == 3
    assert summary["faults_detected"] == 9
    assert summary["controls_accepted"] == 3
    assert summary["finite_faults"] == 9
    assert all(result.expected_outcome_met for result in results)
    assert all(result.manifest_hash for result in results)


def test_heldout_campaign_is_deterministic_and_writes_row_level_csv(tmp_path):
    first = run_heldout_mutations()
    second = run_heldout_mutations()

    assert first == second

    output = tmp_path / "heldout_mutations.csv"
    written = write_heldout_results(first, output)

    assert written == output
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(first)
    assert {row["family"] for row in rows} == {"mie", "thinfilm", "beam"}
    assert all(row["manifest_hash"] for row in rows)


def test_heldout_default_output_is_an_artifact_path():
    assert "optcon-artifacts" in DEFAULT_OUTPUT_PATH.parts
    assert "private-paper" not in DEFAULT_OUTPUT_PATH.parts
