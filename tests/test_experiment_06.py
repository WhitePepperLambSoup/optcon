"""Regression tests for the Mie decision-impact case study."""

from __future__ import annotations

import csv

import pytest

from optcon.examples.experiment_06_mie_decision_impact import run_decision_impact


def test_mie_medium_convention_changes_the_first_threshold_crossing(tmp_path):
    output_path = tmp_path / "mie_decision_impact.csv"

    result = run_decision_impact(output_path)

    assert result["correct_decision_ready"] is True
    assert result["fault_decision_ready"] is False
    assert result["fault_finite"] is True
    assert result["correct_threshold_diameter_nm"] == pytest.approx(495.65, abs=0.2)
    assert result["fault_threshold_diameter_nm"] == pytest.approx(221.73, abs=0.2)
    assert output_path.is_file()


def test_mie_decision_impact_records_the_reference_block(tmp_path):
    output_path = tmp_path / "mie_decision_impact.csv"

    run_decision_impact(output_path)

    with output_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert [row["case_id"] for row in rows] == [
        "declared-water-medium",
        "vacuum-default-mismatch",
    ]
    assert rows[0]["decision_ready"] == "True"
    assert rows[1]["decision_ready"] == "False"
    assert rows[1]["finite_result"] == "True"
    assert rows[1]["failed_requirements"] == "independent_reference"
    assert rows[0]["status"] == "ready"
    assert rows[1]["status"] == "blocked"
    assert rows[1]["required_evidence"] == (
        "semantic;numerical;independent_reference;provenance;scope"
    )
    assert rows[1]["available_evidence"] == "semantic;numerical;provenance;scope"
