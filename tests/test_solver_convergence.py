"""Regression tests for the reproducible solver-convergence benchmark."""

from __future__ import annotations

import csv
import math

import pytest

from optcon.benchmarks.convergence import _estimated_orders, run_solver_convergence


def test_estimated_orders_hide_machine_precision_plateaus():
    orders = _estimated_orders(
        [2.2e-4, 3.1e-14, 3.3e-14, 5.1e-14],
        [2.0, 2.0, 2.0, 2.0],
    )

    assert orders[0] is None
    assert orders[1:] == [None, None, None]


def test_estimated_orders_keep_resolved_decrease():
    orders = _estimated_orders(
        [1.6e-3, 4.0e-4, 1.0e-4],
        [2.0, 2.0, 2.0],
    )

    assert orders[0] is None
    assert orders[1:] == [pytest.approx(2.0), pytest.approx(2.0)]


def test_solver_convergence_returns_both_solver_families_and_writes_csv(tmp_path):
    output = tmp_path / "solver_convergence.csv"

    rows = run_solver_convergence(
        output_path=output,
        fox_points=(16, 32, 64),
        gnlse_steps=(8, 16, 32),
        include_generalized=False,
    )

    assert {row["solver"] for row in rows} == {"fox_li", "gnlse"}
    assert output.exists()

    with output.open(newline="", encoding="utf-8") as stream:
        written = list(csv.DictReader(stream))
    assert len(written) == len(rows)
    assert {
        "solver",
        "discretization",
        "discretization_value",
        "reference_discretization",
        "relative_error",
        "energy_drift",
        "monitored_invariant",
        "invariant_drift",
        "estimated_order",
        "runtime_s",
    } <= set(written[0])


def test_fox_li_error_decreases_with_grid_refinement():
    rows = run_solver_convergence(
        fox_points=(16, 32, 64),
        gnlse_steps=(8,),
        include_generalized=False,
    )
    fox_rows = [row for row in rows if row["solver"] == "fox_li"]
    errors = [float(row["relative_error"]) for row in fox_rows]
    assert all(math.isfinite(error) for error in errors)
    assert errors[-1] < errors[0]


def test_conservative_gnlse_error_decreases_and_energy_is_tracked():
    rows = run_solver_convergence(
        fox_points=(16,),
        gnlse_steps=(8, 16, 32),
        include_generalized=False,
    )
    gnlse_rows = [row for row in rows if row["solver"] == "gnlse"]
    errors = [float(row["relative_error"]) for row in gnlse_rows]
    drifts = [float(row["energy_drift"]) for row in gnlse_rows]
    assert errors[-1] < errors[0]
    assert max(drifts) < 1e-10
    resolved_orders = [
        float(row["estimated_order"])
        for row in gnlse_rows
        if row["estimated_order"] is not None
    ]
    assert resolved_orders
    assert all(math.isfinite(order) for order in resolved_orders)


def test_generalized_gnlse_reports_energy_and_monitors_photon_number():
    rows = run_solver_convergence(
        fox_points=(16,),
        gnlse_steps=(8, 16),
        include_generalized=True,
    )
    generalized_rows = [
        row
        for row in rows
        if row["solver"] == "gnlse"
        and row["model_variant"] == "raman_self_steepening"
    ]

    assert generalized_rows
    assert {row["monitored_invariant"] for row in generalized_rows} == {
        "photon_number"
    }
    assert all(float(row["energy_drift"]) > 0.0 for row in generalized_rows)
    assert max(float(row["invariant_drift"]) for row in generalized_rows) < 1e-4
