"""Deterministic multi-parameter stress tests for real-domain adjoints.

The benchmark reuses a nonuniform Gauss--Legendre aperture discretisation and
tests a three-parameter phase-and-amplitude map at several parameter points and
probe seeds.  The correct reverse map uses the quadrature metric carried by
the forward representation.  The negative control deliberately uses a
uniform metric in the reverse map.  This is a robustness check for the
implemented dot-product contract, not a population estimate for arbitrary
adjoint implementations.

Run from the repository root with::

    python -m optcon.benchmarks.adjoint_stress
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from optcon.checks import dot_test

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "optcon-artifacts"
    / "benchmarks"
    / "adjoint_stress.csv"
)
DEFAULT_PARAMETER_COUNT = 24
DEFAULT_PROBE_SEEDS = tuple(range(8))
_PARAMETER_SEED = 20260918
_DOT_RTOL = 1e-5
_DOT_EPS = 1e-7


@dataclass(frozen=True)
class AdjointStressRecord:
    """One parameter point and probe comparison in the stress sweep."""

    parameter_index: int
    probe_seed: int
    phase_tilt: float
    phase_defocus: float
    amplitude_perturbation: float
    correct_rel_error: float
    wrong_rel_error: float
    correct_ok: bool
    wrong_detected: bool


def _build_problem() -> tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray, bool], Callable[[np.ndarray], np.ndarray]],
]:
    """Construct the weighted forward map and two metric choices for its adjoint."""
    nodes, normalized_weights = np.polynomial.legendre.leggauss(128)
    half_width = 2.0e-3
    weights = half_width * normalized_weights
    quadrature_metric = np.sqrt(weights)
    uniform_metric = np.full(nodes.size, math_sqrt(2.0 * half_width / nodes.size))

    base_field = np.exp(-(nodes / 0.72) ** 2).astype(complex)
    base_field /= math_sqrt(float(np.sum(normalized_weights * np.abs(base_field) ** 2)))

    def physical_field(parameters: np.ndarray) -> np.ndarray:
        phase_tilt, phase_defocus, amplitude_perturbation = np.asarray(
            parameters, dtype=float
        )
        phase = 0.35 * phase_tilt * nodes + 0.25 * phase_defocus * (
            2.0 * nodes**2 - 1.0
        )
        envelope = 1.0 + 0.05 * amplitude_perturbation
        return envelope * base_field * np.exp(-1.0j * phase)

    def forward(parameters: np.ndarray) -> np.ndarray:
        return quadrature_metric * physical_field(parameters)

    def adjoint_at(
        parameters: np.ndarray, use_quadrature_metric: bool
    ) -> Callable[[np.ndarray], np.ndarray]:
        phase_tilt, phase_defocus, amplitude_perturbation = np.asarray(
            parameters, dtype=float
        )
        phase = 0.35 * phase_tilt * nodes + 0.25 * phase_defocus * (
            2.0 * nodes**2 - 1.0
        )
        envelope = 1.0 + 0.05 * amplitude_perturbation
        field = envelope * base_field * np.exp(-1.0j * phase)
        jacobian_physical = np.column_stack(
            (
                -1.0j * 0.35 * nodes * field,
                -1.0j * 0.25 * (2.0 * nodes**2 - 1.0) * field,
                0.05 * base_field * np.exp(-1.0j * phase),
            )
        )
        metric = quadrature_metric if use_quadrature_metric else uniform_metric
        jacobian = metric[:, None] * jacobian_physical

        def adjoint(probe: np.ndarray) -> np.ndarray:
            return np.real(jacobian.conj().T @ np.asarray(probe))

        return adjoint

    return forward, adjoint_at


def math_sqrt(value: float) -> float:
    """Return a scalar square root without importing a second numerical stack."""
    return float(np.sqrt(value))


def run_adjoint_stress(
    output_path: str | Path | None = None,
    *,
    parameter_count: int = DEFAULT_PARAMETER_COUNT,
    probe_seeds: Sequence[int] = DEFAULT_PROBE_SEEDS,
) -> list[AdjointStressRecord]:
    """Run the multi-parameter adjoint stress sweep and optionally write CSV."""
    if parameter_count < 1:
        raise ValueError("parameter_count must be positive")
    seeds = tuple(int(seed) for seed in probe_seeds)
    if not seeds:
        raise ValueError("probe_seeds must contain at least one seed")

    forward, adjoint_at = _build_problem()
    generator = np.random.default_rng(_PARAMETER_SEED)
    parameter_points = generator.uniform(-1.0, 1.0, size=(parameter_count, 3))
    records: list[AdjointStressRecord] = []

    for parameter_index, point in enumerate(parameter_points):
        correct_adjoint = adjoint_at(point, True)
        wrong_adjoint = adjoint_at(point, False)
        for probe_seed in seeds:
            correct_report = dot_test(
                forward,
                correct_adjoint,
                point,
                seed=probe_seed,
                eps=_DOT_EPS,
                rtol=_DOT_RTOL,
            )
            wrong_report = dot_test(
                forward,
                wrong_adjoint,
                point,
                seed=probe_seed,
                eps=_DOT_EPS,
                rtol=_DOT_RTOL,
            )
            records.append(
                AdjointStressRecord(
                    parameter_index=parameter_index,
                    probe_seed=probe_seed,
                    phase_tilt=float(point[0]),
                    phase_defocus=float(point[1]),
                    amplitude_perturbation=float(point[2]),
                    correct_rel_error=float(correct_report["rel_error"]),
                    wrong_rel_error=float(wrong_report["rel_error"]),
                    correct_ok=bool(correct_report["ok"]),
                    wrong_detected=not bool(wrong_report["ok"]),
                )
            )

    if output_path is not None:
        write_adjoint_stress_results(records, output_path)
    return records


def write_adjoint_stress_results(
    records: Sequence[AdjointStressRecord], output_path: str | Path = DEFAULT_OUTPUT_PATH
) -> Path:
    """Write row-level stress measurements to an auditable CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(AdjointStressRecord.__annotations__)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return path


def summarize_adjoint_stress(records: Sequence[AdjointStressRecord]) -> dict[str, Any]:
    """Summarize pass and detection counts without inferring population rates."""
    if not records:
        raise ValueError("records must not be empty")
    return {
        "records": len(records),
        "correct_passes": sum(record.correct_ok for record in records),
        "wrong_detections": sum(record.wrong_detected for record in records),
        "correct_max_rel_error": max(record.correct_rel_error for record in records),
        "wrong_min_rel_error": min(record.wrong_rel_error for record in records),
    }


def main() -> int:
    records = run_adjoint_stress(output_path=DEFAULT_OUTPUT_PATH)
    summary = summarize_adjoint_stress(records)
    print(
        "adjoint stress: "
        f"correct {summary['correct_passes']}/{summary['records']}, "
        f"wrong detected {summary['wrong_detections']}/{summary['records']}"
    )
    print(
        "relative-error range: "
        f"correct max {summary['correct_max_rel_error']:.3e}, "
        f"wrong min {summary['wrong_min_rel_error']:.3e}"
    )
    print(f"wrote {DEFAULT_OUTPUT_PATH}")
    return int(
        summary["correct_passes"] != summary["records"]
        or summary["wrong_detections"] != summary["records"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
