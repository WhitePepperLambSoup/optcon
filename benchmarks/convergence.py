"""Deterministic discretization-convergence measurements for the paper.

The benchmark deliberately uses a finer numerical result as a reference rather
than an analytic expression whose geometry, boundary convention, or nonlinear
model might differ from the implementation under test.  It is an evidence
generator, not a replacement for unit tests or a proof of asymptotic order.

Run from the repository root with::

    python -m optcon.benchmarks.convergence
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from optcon import q
from optcon.fox_li import FoxLiResonator, solve_fox_li_modes
from optcon.nlse import FiberParameters, gaussian_pulse, solve_nlse

DEFAULT_FOX_POINTS = (16, 32, 64, 128, 256)
DEFAULT_GNLSE_STEPS = (25, 50, 100, 200)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "data" / "solver_convergence.csv"
)

CSV_FIELDS = [
    "solver",
    "model_variant",
    "discretization",
    "discretization_value",
    "reference_discretization",
    "metric",
    "metric_value",
    "reference_metric_value",
    "one_way_loss",
    "round_trip_loss",
    "relative_error",
    "relative_loss_error",
    "energy_drift",
    "estimated_order",
    "runtime_s",
    "samples",
    "time_window_ps",
]

# Relative errors below this level are treated as a numerical floor rather than
# evidence of a resolved asymptotic regime.  The threshold is deliberately
# conservative because the reference solution is itself a finite discretization.
DEFAULT_ORDER_ERROR_FLOOR = 1e-12


def _relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-15)


def _phase_aligned_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Return relative field error after removing the arbitrary global phase."""
    denominator = max(float(np.linalg.norm(reference)), 1e-30)
    overlap = np.vdot(reference, candidate)
    if abs(overlap) > 0.0:
        candidate = candidate * np.exp(-1.0j * np.angle(overlap))
    return float(np.linalg.norm(candidate - reference) / denominator)


def _estimated_orders(
    errors: list[float],
    refinements: list[float],
    *,
    error_floor: float = DEFAULT_ORDER_ERROR_FLOOR,
) -> list[float | None]:
    """Estimate refinement orders only while the error is numerically resolved.

    ``None`` means that no meaningful order is available.  In particular, an
    error at or below ``error_floor`` is reported as a platform value instead
    of being converted into a spurious very large order.
    """
    if len(errors) != len(refinements):
        raise ValueError("errors and refinements must have the same length")
    if error_floor < 0.0:
        raise ValueError("error_floor must be non-negative")

    orders: list[float | None] = [None]
    for previous, current, ratio in zip(errors, errors[1:], refinements, strict=False):
        if (
            not math.isfinite(previous)
            or not math.isfinite(current)
            or previous <= error_floor
            or current <= error_floor
            or ratio <= 1.0
        ):
            orders.append(None)
        else:
            order = float(math.log(previous / current) / math.log(ratio))
            orders.append(order if math.isfinite(order) else None)
    return orders


def _fox_li_rows(points: tuple[int, ...]) -> list[dict[str, Any]]:
    resonator = FoxLiResonator(
        wavelength=q(1064.0, "nm"),
        length=q(500.0, "mm"),
        aperture=q(1.0, "mm"),
        g1=1.0,
        g2=1.0,
        geometry="strip",
    )
    reference_points = max(points) * 2
    reference = solve_fox_li_modes(
        resonator,
        num_modes=1,
        num_points=reference_points,
        check_contracts=False,
    )[0]

    measurements: list[tuple[int, Any, float]] = []
    for num_points in points:
        start = time.perf_counter()
        mode = solve_fox_li_modes(
            resonator,
            num_modes=1,
            num_points=num_points,
            check_contracts=False,
        )[0]
        measurements.append((num_points, mode, time.perf_counter() - start))

    eigen_errors = [
        _relative_error(abs(mode.eigenvalue), abs(reference.eigenvalue))
        for _, mode, _ in measurements
    ]
    loss_errors = [
        _relative_error(mode.loss, reference.loss) for _, mode, _ in measurements
    ]
    orders = _estimated_orders(eigen_errors, [2.0] * len(eigen_errors))

    rows: list[dict[str, Any]] = []
    for (num_points, mode, runtime_s), eigen_error, loss_error, order in zip(
        measurements, eigen_errors, loss_errors, orders, strict=True
    ):
        rows.append(
            {
                "solver": "fox_li",
                "model_variant": "flat_flat_strip",
                "discretization": "num_points",
                "discretization_value": num_points,
                "reference_discretization": reference_points,
                "metric": "fundamental_eigenvalue_magnitude",
                "metric_value": abs(mode.eigenvalue),
                "reference_metric_value": abs(reference.eigenvalue),
                "one_way_loss": mode.loss,
                "round_trip_loss": mode.round_trip_loss,
                "relative_error": eigen_error,
                "relative_loss_error": loss_error,
                "energy_drift": 0.0,
                "estimated_order": order,
                "runtime_s": runtime_s,
                "samples": "",
                "time_window_ps": "",
            }
        )
    return rows


def _gnlse_rows(steps: tuple[int, ...], *, generalized: bool) -> list[dict[str, Any]]:
    samples = 128
    time_window_ps = 2.0
    if generalized:
        pulse = gaussian_pulse(
            peak_power=q(5.0, "W"),
            fwhm_duration=q(200.0, "fs"),
            wavelength=q(1550.0, "nm"),
            samples=samples,
            time_window=q(time_window_ps, "ps"),
        )
        fiber = FiberParameters(
            beta2=q(-20.0, "ps^2/km"),
            gamma=q(2.0, "1/(W*km)"),
            raman_fraction=0.18,
            self_steepening=True,
        )
        distance = q(1.0, "m")
        variant = "raman_self_steepening"
    else:
        pulse = gaussian_pulse(
            peak_power=q(5.0, "W"),
            fwhm_duration=q(200.0, "fs"),
            wavelength=q(1550.0, "nm"),
            samples=samples,
            time_window=q(time_window_ps, "ps"),
        )
        fiber = FiberParameters(
            beta2=q(-20.0, "ps^2/km"),
            gamma=q(2.0, "1/(W*km)"),
        )
        distance = q(10.0, "m")
        variant = "conservative_gvd_spm"

    reference_steps = max(steps) * 2
    reference_pulse, reference_history = solve_nlse(
        pulse,
        fiber,
        distance=distance,
        steps=reference_steps,
        check_energy=not generalized,
    )

    measurements: list[tuple[int, Any, list[float], float]] = []
    for step_count in steps:
        start = time.perf_counter()
        result, history = solve_nlse(
            pulse,
            fiber,
            distance=distance,
            steps=step_count,
            check_energy=not generalized,
        )
        measurements.append((step_count, result, history, time.perf_counter() - start))

    errors = [
        _phase_aligned_l2(result.amplitude, reference_pulse.amplitude)
        for _, result, _, _ in measurements
    ]
    orders = _estimated_orders(errors, [2.0] * len(errors))

    rows: list[dict[str, Any]] = []
    for (step_count, _result, history, runtime_s), error, order in zip(
        measurements, errors, orders, strict=True
    ):
        drift = max(abs(float(value) - 1.0) for value in history)
        reference_drift = max(abs(float(value) - 1.0) for value in reference_history)
        rows.append(
            {
                "solver": "gnlse",
                "model_variant": variant,
                "discretization": "steps",
                "discretization_value": step_count,
                "reference_discretization": reference_steps,
                "metric": "phase_aligned_relative_l2_field_error",
                "metric_value": error,
                "reference_metric_value": 0.0,
                "one_way_loss": "",
                "round_trip_loss": "",
                "relative_error": error,
                "relative_loss_error": "",
                "energy_drift": drift,
                "estimated_order": order,
                "runtime_s": runtime_s,
                "samples": samples,
                "time_window_ps": time_window_ps,
                "reference_energy_drift": reference_drift,
            }
        )
    return rows


def _write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [*CSV_FIELDS, "reference_energy_drift"]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_solver_convergence(
    output_path: str | Path | None = None,
    *,
    fox_points: tuple[int, ...] = DEFAULT_FOX_POINTS,
    gnlse_steps: tuple[int, ...] = DEFAULT_GNLSE_STEPS,
    include_generalized: bool = True,
) -> list[dict[str, Any]]:
    """Run and optionally persist Fox--Li and G-NLSE convergence measurements."""
    if not fox_points or any(point < 16 for point in fox_points):
        raise ValueError("fox_points must contain values >= 16")
    if not gnlse_steps or any(step < 1 for step in gnlse_steps):
        raise ValueError("gnlse_steps must contain positive values")

    ordered_points = tuple(sorted(set(fox_points)))
    ordered_steps = tuple(sorted(set(gnlse_steps)))
    rows = _fox_li_rows(ordered_points)
    rows.extend(_gnlse_rows(ordered_steps, generalized=False))
    if include_generalized:
        rows.extend(_gnlse_rows(ordered_steps, generalized=True))

    if output_path is not None:
        _write_rows(rows, Path(output_path))
    return rows


def main() -> int:
    output = run_solver_convergence(output_path=DEFAULT_OUTPUT_PATH)
    print(f"Wrote {len(output)} convergence rows to {DEFAULT_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
