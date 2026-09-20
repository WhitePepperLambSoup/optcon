"""Regression tests for the thin-film decision-impact case study."""

from __future__ import annotations

import csv

import pytest

from optcon.engines import available_engines
from optcon.examples.experiment_07_thinfilm_decision_impact import run_decision_impact

pytestmark = pytest.mark.skipif(
    "tmm_core" not in available_engines("thin_film"),
    reason="tmm_core unavailable",
)


def test_thinfilm_stack_convention_changes_the_pair_count(tmp_path):
    output_path = tmp_path / "thinfilm_decision_impact.csv"

    result = run_decision_impact(output_path)

    assert result["correct_decision_ready"] is True
    assert result["fault_decision_ready"] is False
    assert result["fault_finite"] is True
    assert result["correct_pairs"] == 6
    assert result["fault_pairs"] == 7
    assert result["pair_distortion"] == 1
    assert result["correct_max_reference_error"] < 1e-12
    assert result["fault_max_reference_error"] > 1e-3
    assert output_path.is_file()


def test_thinfilm_decision_impact_records_the_reference_block(tmp_path):
    output_path = tmp_path / "thinfilm_decision_impact.csv"

    run_decision_impact(output_path)

    with output_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert [row["case_id"] for row in rows] == [
        "declared-terminal-high",
        "omitted-terminal-high",
    ]
    assert rows[0]["candidate_stack"] == "(HL)^N H"
    assert rows[1]["candidate_stack"] == "(HL)^N"
    assert rows[1]["reference_stack"] == "(HL)^N H"
    assert rows[0]["decision_ready"] == "True"
    assert rows[1]["decision_ready"] == "False"
    assert rows[1]["finite_result"] == "True"
    assert rows[1]["failed_requirements"] == "independent_reference"
    assert rows[1]["required_evidence"] == (
        "semantic;numerical;independent_reference;provenance;scope"
    )
    assert rows[1]["available_evidence"] == "semantic;numerical;provenance;scope"
