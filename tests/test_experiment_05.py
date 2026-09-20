"""Reproducibility contracts for the cavity tolerance case study."""

import csv
import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from optcon import q
from optcon.examples.experiment_05_cavity_thermal_tolerance import (
    _thermal_point,
    run_decision_impact,
    run_experiment,
)


def _reference_cavity_q(diopter: float) -> complex:
    # Explicit unfolded half-trip: midpoint -> mirror -> midpoint -> lens.
    space = np.array([[1.0, 0.05], [0.0, 1.0]])
    mirror = np.array([[1.0, 0.0], [-10.0, 1.0]])
    lens = np.array([[1.0, 0.0], [-diopter, 1.0]])
    half_trip = lens @ space @ mirror @ space
    a, b = half_trip[0]
    c, d = half_trip[1]
    roots = np.roots([c, d - a, -b])
    return complex(next(root for root in roots if root.imag > 0.0))


def _reference_field_overlap(cavity_q: complex, tilt_rad: float) -> float:
    wavelength = 1.064e-6
    nominal_z_r = math.sqrt(0.10 * (0.40 - 0.10)) / 2.0
    nominal_waist = math.sqrt(wavelength * nominal_z_r / math.pi)
    inverse_q = 1.0 / cavity_q
    radius = math.sqrt(-wavelength / (math.pi * inverse_q.imag))
    k = 2.0 * math.pi / wavelength
    axis = np.linspace(-1.6e-3, 1.6e-3, 1025)
    product = np.exp(
        -(axis / nominal_waist) ** 2
        - (axis / radius) ** 2
        + 0.5j * k * inverse_q.real * axis**2
    )
    integral_x = trapezoid(product * np.exp(1j * k * tilt_rad * axis), axis)
    integral_y = trapezoid(product, axis)
    amplitude = 2.0 * integral_x * integral_y / (math.pi * nominal_waist * radius)
    return float(abs(amplitude) ** 2)


@pytest.mark.parametrize("diopter", [-10.0, -5.0, 5.0, 15.0])
def test_thermal_coupling_matches_a_two_pass_same_plane_field_integral(diopter):
    wavelength = 1.064e-6
    nominal_waist = math.sqrt(wavelength * math.sqrt(0.03) / (2.0 * math.pi))
    point = _thermal_point(
        diopter,
        length_m=0.10,
        radius_c=q(200.0, "mm"),
        wavelength_m=wavelength,
        nominal_waist_m=nominal_waist,
    )

    assert point.stable
    assert point.coupling == pytest.approx(
        _reference_field_overlap(_reference_cavity_q(diopter), 0.0), abs=1e-12
    )


def test_thermal_lens_rejects_the_unstable_defocusing_case():
    point = _thermal_point(
        -16.0,
        length_m=0.10,
        radius_c=q(200.0, "mm"),
        wavelength_m=1.064e-6,
        nominal_waist_m=171.26207179769648e-6,
    )

    assert not point.stable
    assert point.coupling == 0.0


def test_joint_tilt_thermal_map_matches_direct_field_overlap(tmp_path):
    data_dir = tmp_path / "data"
    run_experiment(output_dir=tmp_path, data_dir=data_dir)
    with (data_dir / "cavity_tolerance_map.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    row = min(
        rows,
        key=lambda record: abs(float(record["diopter_d_m"]) - 5.0)
        + abs(float(record["tilt_urad"]) - 1000.0) / 100.0,
    )
    diopter = float(row["diopter_d_m"])
    tilt_rad = float(row["tilt_urad"]) * 1e-6
    expected = _reference_field_overlap(_reference_cavity_q(diopter), tilt_rad)

    assert float(row["coupling"]) == pytest.approx(expected, abs=1e-12)


def test_experiment_05_returns_tolerance_metrics():
    result = run_experiment()

    expected = {
        "max_tilt_discrepancy",
        "tilt_90pct_threshold_urad",
        "thermal_90pct_negative_limit_d_m",
        "thermal_90pct_positive_limit_d_m",
        "min_thermal_coupling",
        "nominal_waist_um",
        "finesse",
    }
    assert expected <= result.keys()
    assert result["max_tilt_discrepancy"] < 1e-12
    assert 600.0 < result["tilt_90pct_threshold_urad"] < 700.0
    assert -9.0 < result["thermal_90pct_negative_limit_d_m"] < -8.0
    assert 16.0 < result["thermal_90pct_positive_limit_d_m"] < 17.0
    assert 0.0 <= result["min_thermal_coupling"] <= 1.0


def test_experiment_05_reports_a_conservative_90_percent_thermal_budget():
    result = run_experiment()
    negative = abs(result["thermal_90pct_negative_limit_d_m"])
    positive = result["thermal_90pct_positive_limit_d_m"]

    # Use the narrower side of the asymmetric interval as the conservative budget.
    assert min(negative, positive) == result["thermal_90pct_budget_abs_d_m"]
    assert 0.0 < result["thermal_90pct_budget_abs_d_m"] < 10.0
    assert math.isfinite(result["thermal_90pct_budget_abs_d_m"])

def test_experiment_05_supports_an_explicit_output_directory(tmp_path):
    data_dir = tmp_path / "data"
    result = run_experiment(output_dir=tmp_path, data_dir=data_dir)

    assert result["max_tilt_discrepancy"] < 1e-12
    assert (tmp_path / "fig4_cavity_tolerance.png").is_file()
    expected_data = {
        "cavity_tilt_sweep.csv",
        "cavity_thermal_sweep.csv",
        "cavity_tolerance_map.csv",
        "cavity_summary.csv",
    }
    assert {path.name for path in data_dir.iterdir()} == expected_data

    tilt_rows = (data_dir / "cavity_tilt_sweep.csv").read_text(encoding="utf-8")
    assert "tilt_urad,tem00_numeric,tem00_analytical" in tilt_rows.splitlines()[0]
    assert len(tilt_rows.splitlines()) == 32

    map_rows = (data_dir / "cavity_tolerance_map.csv").read_text(encoding="utf-8")
    assert len(map_rows.splitlines()) == 3601


def test_cavity_decision_impact_records_contract_and_budget_change(tmp_path):
    output_path = tmp_path / "decision_impact.csv"
    result = run_decision_impact(output_path)

    assert result["correct_contract"] is True
    assert result["fault_contract"] is False
    assert result["fault_budget_reported"] is False
    assert float(result["correct_budget_urad"]) == pytest.approx(640.43, abs=0.2)
    assert 400.0 < float(result["fault_unchecked_candidate_budget_urad"]) < 550.0
    assert result["fault_diagnostic_type"] == "AmplitudeOrderError"
    assert output_path.is_file()

    rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))
    assert [row["case_id"] for row in rows] == [
        "correct-field-to-power",
        "fault-power-used-as-field",
    ]
    assert rows[0]["reported_budget_urad"]
    assert rows[1]["reported_budget_urad"] == ""
    assert rows[1]["unchecked_candidate_budget_urad"]


def test_cavity_decision_impact_reuses_the_evidence_promotion_gate(tmp_path):
    output_path = tmp_path / "decision_impact.csv"
    result = run_decision_impact(output_path)

    assert result["correct_decision_ready"] is True
    assert result["fault_decision_ready"] is False

    with output_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert rows[0]["decision_ready"] == "True"
    assert rows[1]["decision_ready"] == "False"
    assert rows[1]["failed_requirements"] == "semantic"
    assert rows[0]["status"] == "ready"
    assert rows[1]["status"] == "blocked"
    assert rows[1]["required_evidence"] == "semantic;numerical;provenance;scope"
    assert rows[1]["available_evidence"] == "numerical;provenance;scope"
