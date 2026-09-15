"""Reproducibility contracts for the cavity tolerance case study."""

import math

from optcon.examples.experiment_05_cavity_thermal_tolerance import run_experiment


def test_experiment_05_returns_manuscript_tolerance_metrics():
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
    assert result["thermal_90pct_negative_limit_d_m"] < 0.0
    assert result["thermal_90pct_positive_limit_d_m"] > 0.0
    assert 0.0 <= result["min_thermal_coupling"] <= 1.0


def test_experiment_05_reports_a_conservative_90_percent_thermal_budget():
    result = run_experiment()
    negative = abs(result["thermal_90pct_negative_limit_d_m"])
    positive = result["thermal_90pct_positive_limit_d_m"]

    # The published |D_th| < 8 m^-1 statement is a conservative summary of
    # the asymmetric sampled curve, not a hard-coded value in the experiment.
    assert min(negative, positive) == result["thermal_90pct_budget_abs_d_m"]
    assert 0.0 < result["thermal_90pct_budget_abs_d_m"] < 10.0
    assert math.isfinite(result["thermal_90pct_budget_abs_d_m"])

def test_experiment_05_supports_an_explicit_output_directory(tmp_path):
    result = run_experiment(output_dir=tmp_path)

    assert result["max_tilt_discrepancy"] < 1e-12
    assert (tmp_path / "fig4_cavity_tolerance.png").is_file()
