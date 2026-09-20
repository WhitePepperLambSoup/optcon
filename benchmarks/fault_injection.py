"""Deterministic fault-injection benchmark for optical semantic contracts.

The benchmark evaluates deliberately silent defects through three layers:

* a numerical smoke test that checks only whether the computation finishes
  with finite values;
* a dimension-only baseline that enforces units but does not track optical
  amplitude provenance or operator semantics;
* the complete optcon contract layer.

The cases are small enough to run in continuous integration. They are not a
population estimate of all scientific-software defects; they test whether the
implemented contracts detect the specific failure classes they claim to cover.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from optcon import (
    amplitude_ratio,
    assert_adjoint,
    assert_passive,
    assert_reciprocal,
    assert_unitary,
    power_ratio,
    q,
    sqrt,
)
from optcon.checks import dot_test
from optcon.engines.differential import assert_engines_agree
from optcon.engines.reference_mie import mie_efficiencies_reference

DEFAULT_FAULT_INJECTION_PATH = (
    Path(__file__).resolve().parents[1]
    / "optcon-artifacts"
    / "benchmarks"
    / "fault_injection.csv"
)


@dataclass(frozen=True)
class FaultInjectionResult:
    """One benchmark case evaluated through all three detection layers."""

    case_id: str
    family: str
    description: str
    is_fault: bool
    effect_metric: str
    effect_value: float
    numeric_smoke_detected: bool
    dimension_only_detected: bool
    full_contract_detected: bool
    diagnostic_type: str
    diagnostic_message: str
    numeric_smoke_runtime_us: float
    dimension_only_runtime_us: float
    full_contract_runtime_us: float


@dataclass(frozen=True)
class _Case:
    case_id: str
    family: str
    description: str
    is_fault: bool
    effect_metric: str
    effect_value: float
    numeric_operation: Callable[[], Any]
    dimension_operation: Callable[[], Any]
    contract_operation: Callable[[], Any]


def _walk_numeric_values(value: Any):
    if value is None:
        return
    if hasattr(value, "value"):
        yield from _walk_numeric_values(value.value)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_numeric_values(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_numeric_values(item)
        return
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.number):
        yield array


def _numeric_smoke(operation: Callable[[], Any]) -> None:
    result = operation()
    for array in _walk_numeric_values(result):
        if not np.isfinite(array).all():
            raise FloatingPointError("operation returned non-finite numerical values")


def _run_numeric_case(case: _Case) -> None:
    _numeric_smoke(case.numeric_operation)


def _measure_detection(
    operation: Callable[[], Any], repeats: int
) -> tuple[bool, str, str, float]:
    elapsed: list[float] = []
    detection: tuple[bool, str, str] | None = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        try:
            operation()
            current = (False, "", "")
        except Exception as error:
            current = (True, type(error).__name__, str(error).splitlines()[0])
        elapsed.append((time.perf_counter_ns() - started) / 1000.0)
        if detection is None:
            detection = current
        elif current != detection:
            raise RuntimeError("fault detector produced a non-deterministic outcome")
    assert detection is not None
    return (*detection, float(median(elapsed)))


def _fft_matrix(size: int, *, normalized: bool) -> np.ndarray:
    indices = np.arange(size)
    matrix = np.exp(-2.0j * math.pi * np.outer(indices, indices) / size)
    return matrix / math.sqrt(size) if normalized else matrix


def _complex_linear_pair(*, conjugate: bool) -> tuple[Callable, Callable, np.ndarray]:
    matrix = np.array(
        [[1.0 + 0.5j, -0.25 + 0.75j], [0.2 - 0.4j, 0.9 + 0.1j]],
        dtype=complex,
    )

    def forward(vector: np.ndarray) -> np.ndarray:
        return matrix @ vector

    def adjoint(vector: np.ndarray) -> np.ndarray:
        operator = matrix.conj().T if conjugate else matrix.T
        return operator @ vector

    return forward, adjoint, np.array([0.4 + 0.1j, -0.2 + 0.3j])


def _real_to_complex_pair(
    *, return_real: bool
) -> tuple[Callable, Callable, np.ndarray]:
    derivative = np.array([[1.0 + 2.0j], [-0.5 + 0.25j]])

    def forward(vector: np.ndarray) -> np.ndarray:
        return derivative @ vector

    def adjoint(vector: np.ndarray) -> np.ndarray:
        cotangent = derivative.conj().T @ vector
        return np.real(cotangent) if return_real else cotangent

    return forward, adjoint, np.array([0.3])


def _quadrature_pair(*, correct_metric: bool) -> tuple[Callable, Callable, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(32)
    derivative = np.sqrt(weights) * np.exp(1.0j * 0.7 * nodes)
    wrong_derivative = np.full(nodes.size, math.sqrt(2.0 / nodes.size)) * np.exp(
        1.0j * 0.7 * nodes
    )

    def forward(parameter: np.ndarray) -> np.ndarray:
        return parameter[0] * derivative

    def adjoint(vector: np.ndarray) -> np.ndarray:
        active = derivative if correct_metric else wrong_derivative
        return np.array([float(np.real(np.vdot(active, vector)))])

    return forward, adjoint, np.array([0.2])


def _mie_query(engine: str) -> dict[str, float]:
    if engine == "vacuum_reference":
        return mie_efficiencies_reference(1.5 + 0.01j, 200.0, 550.0, n_env=1.0)
    if engine == "air_default_fault":
        return mie_efficiencies_reference(
            1.5 + 0.01j,
            200.0,
            550.0,
            n_env=1.00027316,
        )
    raise ValueError(f"unknown synthetic engine {engine!r}")


def _mie_convention_effect() -> float:
    vacuum = _mie_query("vacuum_reference")["Qext"]
    air = _mie_query("air_default_fault")["Qext"]
    return abs(air - vacuum) / max(abs(air), abs(vacuum))


def _cases() -> list[_Case]:
    qwp = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)
    misnormalised_qwp = 0.9 * qwp
    active_mirror = np.array([[1.4]], dtype=complex)
    asymmetric_scatterer = np.array([[0.2, 0.7], [0.3, 0.2]], dtype=complex)
    reciprocal_scatterer = np.array([[0.2, 0.5], [0.5, 0.2]], dtype=complex)
    complex_forward, complex_adjoint, complex_point = _complex_linear_pair(conjugate=True)
    _, missing_conjugate, _ = _complex_linear_pair(conjugate=False)
    real_forward, real_adjoint, real_point = _real_to_complex_pair(return_real=True)
    _, complex_cotangent, _ = _real_to_complex_pair(return_real=False)
    quadrature_forward, quadrature_adjoint, quadrature_point = _quadrature_pair(
        correct_metric=True
    )
    _, uniform_metric_adjoint, _ = _quadrature_pair(correct_metric=False)
    fft_normalized = _fft_matrix(8, normalized=True)
    fft_unnormalized = _fft_matrix(8, normalized=False)
    missing_conjugate_report = dot_test(
        complex_forward,
        missing_conjugate,
        complex_point,
        seed=5,
    )
    complex_cotangent_report = dot_test(
        real_forward,
        complex_cotangent,
        real_point,
        seed=7,
    )
    uniform_metric_report = dot_test(
        quadrature_forward,
        uniform_metric_adjoint,
        quadrature_point,
        seed=11,
    )

    return [
        _Case(
            "control-compatible-lengths",
            "dimensions",
            "Compatible lengths expressed in nanometres and millimetres",
            False,
            "relative_error",
            0.0,
            lambda: 1064.0e-6 + 0.3,
            lambda: q(1064.0, "nm") + q(0.3, "mm"),
            lambda: q(1064.0, "nm") + q(0.3, "mm"),
        ),
        _Case(
            "control-power-to-field",
            "amplitude_provenance",
            "Single square root converts a power transmission to a field amplitude",
            False,
            "relative_error",
            0.0,
            lambda: math.sqrt(0.81),
            lambda: math.sqrt(0.81),
            lambda: sqrt(power_ratio(0.81)),
        ),
        _Case(
            "control-lossless-qwp",
            "unitarity",
            "Lossless quarter-wave plate in an orthonormal Jones basis",
            False,
            "unitarity_residual",
            0.0,
            lambda: qwp @ np.array([1.0, 0.0], dtype=complex),
            lambda: qwp,
            lambda: assert_unitary(qwp, name="quarter-wave plate"),
        ),
        _Case(
            "control-reciprocal-scatterer",
            "reciprocity",
            "Symmetric two-port scattering operator",
            False,
            "reciprocity_residual",
            0.0,
            lambda: reciprocal_scatterer @ np.ones(2),
            lambda: reciprocal_scatterer,
            lambda: assert_reciprocal(reciprocal_scatterer, name="two-port"),
        ),
        _Case(
            "control-complex-adjoint",
            "adjoint_conjugation",
            "Hermitian adjoint for a complex-linear operator",
            False,
            "adjoint_relative_error",
            0.0,
            lambda: complex_forward(complex_point),
            lambda: complex_forward(complex_point),
            lambda: assert_adjoint(
                complex_forward,
                complex_adjoint,
                complex_point,
                seed=5,
                name="complex-linear operator",
            ),
        ),
        _Case(
            "control-real-parameter-adjoint",
            "adjoint_domain",
            "Real cotangent for a real parameter mapped to a complex field",
            False,
            "adjoint_relative_error",
            float(
                dot_test(
                    real_forward,
                    real_adjoint,
                    real_point,
                    seed=7,
                )["rel_error"]
            ),
            lambda: real_adjoint(real_forward(real_point)),
            lambda: real_adjoint(real_forward(real_point)),
            lambda: assert_adjoint(
                real_forward,
                real_adjoint,
                real_point,
                seed=7,
                name="real-parameter operator",
            ),
        ),
        _Case(
            "control-normalized-fft",
            "fft_normalization",
            "Unitary discrete Fourier transform with orthonormal scaling",
            False,
            "unitarity_residual",
            0.0,
            lambda: fft_normalized @ np.ones(8),
            lambda: fft_normalized,
            lambda: assert_unitary(fft_normalized, name="normalized DFT"),
        ),
        _Case(
            "fault-angle-added-to-reflectance",
            "dimensions",
            "A mirror tilt is added to a scalar reflectance",
            True,
            "relative_output_error",
            abs((0.9 + 0.0003) - 0.9) / 0.9,
            lambda: 0.9 + 0.0003,
            lambda: q(0.9, "1") + q(0.3, "mrad"),
            lambda: q(0.9, "1") + q(0.3, "mrad"),
        ),
        _Case(
            "fault-field-power-addition",
            "amplitude_provenance",
            "A power transmission is added to a field transmission",
            True,
            "relative_field_error",
            abs((0.9 + 0.81) - (0.9 + math.sqrt(0.81))) / 1.8,
            lambda: 0.9 + 0.81,
            lambda: 0.9 + 0.81,
            lambda: amplitude_ratio(0.9) + power_ratio(0.81),
        ),
        _Case(
            "fault-repeated-square-root",
            "amplitude_provenance",
            "A field transmission is square-rooted a second time",
            True,
            "relative_field_error",
            abs(math.sqrt(0.9) - 0.9) / 0.9,
            lambda: math.sqrt(math.sqrt(0.81)),
            lambda: math.sqrt(math.sqrt(0.81)),
            lambda: sqrt(amplitude_ratio(0.9)),
        ),
        _Case(
            "fault-lossy-declared-unitary",
            "unitarity",
            "A misnormalised Jones operator is declared lossless",
            True,
            "maximum_power_norm_drift",
            1.0 - 0.9**2,
            lambda: misnormalised_qwp @ np.array([1.0, 0.0], dtype=complex),
            lambda: misnormalised_qwp,
            lambda: assert_unitary(misnormalised_qwp, name="misnormalised QWP"),
        ),
        _Case(
            "fault-active-declared-passive",
            "passivity",
            "An amplifying scalar operator is declared passive",
            True,
            "power_gain_above_unity",
            1.4**2 - 1.0,
            lambda: active_mirror @ np.array([1.0], dtype=complex),
            lambda: active_mirror,
            lambda: assert_passive(active_mirror, name="active mirror"),
        ),
        _Case(
            "fault-asymmetric-declared-reciprocal",
            "reciprocity",
            "An asymmetric two-port operator is declared reciprocal",
            True,
            "reciprocity_residual",
            float(np.max(np.abs(asymmetric_scatterer - asymmetric_scatterer.T))),
            lambda: asymmetric_scatterer @ np.ones(2),
            lambda: asymmetric_scatterer,
            lambda: assert_reciprocal(asymmetric_scatterer, name="asymmetric two-port"),
        ),
        _Case(
            "fault-missing-complex-conjugation",
            "adjoint_conjugation",
            "The transpose is used instead of the Hermitian transpose",
            True,
            "adjoint_relative_error",
            float(missing_conjugate_report["rel_error"]),
            lambda: missing_conjugate(complex_forward(complex_point)),
            lambda: missing_conjugate(complex_forward(complex_point)),
            lambda: assert_adjoint(
                complex_forward,
                missing_conjugate,
                complex_point,
                seed=5,
                name="transpose-only adjoint",
            ),
        ),
        _Case(
            "fault-complex-cotangent-for-real-parameter",
            "adjoint_domain",
            "A real parameter is returned a complex cotangent",
            True,
            "adjoint_domain_error",
            float(complex_cotangent_report["adjoint_domain_error"]),
            lambda: complex_cotangent(real_forward(real_point)),
            lambda: complex_cotangent(real_forward(real_point)),
            lambda: assert_adjoint(
                real_forward,
                complex_cotangent,
                real_point,
                seed=7,
                name="real-parameter adjoint",
            ),
        ),
        _Case(
            "fault-omitted-quadrature-metric",
            "quadrature_metric",
            "The adjoint assumes uniform weights on Gauss-Legendre nodes",
            True,
            "adjoint_relative_error",
            float(uniform_metric_report["rel_error"]),
            lambda: uniform_metric_adjoint(quadrature_forward(quadrature_point)),
            lambda: uniform_metric_adjoint(quadrature_forward(quadrature_point)),
            lambda: assert_adjoint(
                quadrature_forward,
                uniform_metric_adjoint,
                quadrature_point,
                seed=11,
                name="uniform-weight adjoint",
            ),
        ),
        _Case(
            "fault-unnormalized-fft-declared-unitary",
            "fft_normalization",
            "An unnormalised DFT is declared unitary",
            True,
            "power_norm_inflation",
            7.0,
            lambda: fft_unnormalized @ np.ones(8),
            lambda: fft_unnormalized,
            lambda: assert_unitary(fft_unnormalized, name="unnormalized DFT"),
        ),
        _Case(
            "fault-mie-medium-convention",
            "engine_convention",
            "Two Mie calculations silently use vacuum and air as different media",
            True,
            "maximum_relative_engine_difference",
            _mie_convention_effect(),
            lambda: _mie_query("air_default_fault"),
            lambda: _mie_query("air_default_fault"),
            lambda: assert_engines_agree(
                _mie_query,
                ("vacuum_reference", "air_default_fault"),
                rtol=1e-6,
                name="Mie surrounding-medium convention",
            ),
        ),
    ]


def run_fault_injection_benchmark(*, repeats: int = 7) -> list[FaultInjectionResult]:
    """Execute all valid controls and injected faults.

    ``repeats`` affects only timing; each detector must return the same logical
    outcome on every repetition.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")

    results: list[FaultInjectionResult] = []
    for case in _cases():
        smoke = _measure_detection(partial(_run_numeric_case, case), repeats)
        dimension = _measure_detection(case.dimension_operation, repeats)
        contract = _measure_detection(case.contract_operation, repeats)
        results.append(
            FaultInjectionResult(
                case_id=case.case_id,
                family=case.family,
                description=case.description,
                is_fault=case.is_fault,
                effect_metric=case.effect_metric,
                effect_value=case.effect_value,
                numeric_smoke_detected=smoke[0],
                dimension_only_detected=dimension[0],
                full_contract_detected=contract[0],
                diagnostic_type=contract[1],
                diagnostic_message=contract[2],
                numeric_smoke_runtime_us=smoke[3],
                dimension_only_runtime_us=dimension[3],
                full_contract_runtime_us=contract[3],
            )
        )
    return results


def summarize_fault_injection(results: list[FaultInjectionResult]) -> dict[str, int]:
    """Return detection and false-positive counts for the three layers."""
    faults = [result for result in results if result.is_fault]
    controls = [result for result in results if not result.is_fault]
    return {
        "faults_total": len(faults),
        "numeric_smoke_detected": sum(
            result.numeric_smoke_detected for result in faults
        ),
        "dimension_only_detected": sum(
            result.dimension_only_detected for result in faults
        ),
        "full_contract_detected": sum(
            result.full_contract_detected for result in faults
        ),
        "controls_total": len(controls),
        "numeric_smoke_false_positives": sum(
            result.numeric_smoke_detected for result in controls
        ),
        "dimension_only_false_positives": sum(
            result.dimension_only_detected for result in controls
        ),
        "full_contract_false_positives": sum(
            result.full_contract_detected for result in controls
        ),
    }


def write_fault_injection_results(
    results: list[FaultInjectionResult],
    output_path: str | Path = DEFAULT_FAULT_INJECTION_PATH,
) -> Path:
    """Write the complete benchmark result table to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        list(asdict(results[0]).keys())
        if results
        else list(FaultInjectionResult.__annotations__)
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    return path


def main() -> int:
    results = run_fault_injection_benchmark()
    summary = summarize_fault_injection(results)
    output = write_fault_injection_results(results)
    print(
        "fault detection: "
        f"numeric {summary['numeric_smoke_detected']}/{summary['faults_total']}, "
        f"dimension-only {summary['dimension_only_detected']}/{summary['faults_total']}, "
        f"full contracts {summary['full_contract_detected']}/{summary['faults_total']}"
    )
    print(
        "false positives: "
        f"numeric {summary['numeric_smoke_false_positives']}/{summary['controls_total']}, "
        f"dimension-only {summary['dimension_only_false_positives']}/{summary['controls_total']}, "
        f"full contracts {summary['full_contract_false_positives']}/{summary['controls_total']}"
    )
    print(f"wrote {output}")
    return int(
        summary["full_contract_detected"] != summary["faults_total"]
        or summary["full_contract_false_positives"] != 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
