"""Cross-engine differential testing.

Independent implementations can provide useful comparative evidence when they
receive the same physical query and expose a common observable.  This module
runs one query through several engines, normalises the outputs through
optcon's unit layer, and reports whether they agree.  A comparison is evidence
about the stated query and numerical regime; it is not an automatic source of
ground truth.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import numpy as np

from ..errors import ContractViolation


def compare_across_engines(
    runner: Callable[[str], Mapping[str, float]],
    engines: Iterable[str],
    rtol: float = 1e-6,
    atol: float = 0.0,
) -> dict:
    """Run ``runner(engine)`` for each engine and compare their numbers.

    All engines are compared against the first one, which acts as the
    reference.  Returns a report rather than raising, so callers can decide
    whether a disagreement is a defect or expected physics.
    """
    names = list(engines)
    if len(names) < 2:
        raise ValueError("differential comparison needs at least two engines")
    if len(set(names)) != len(names):
        raise ValueError("differential comparison needs unique engine names")

    if rtol < 0.0 or atol < 0.0 or not np.isfinite(rtol) or not np.isfinite(atol):
        raise ValueError("rtol and atol must be finite and non-negative")

    values = {name: dict(runner(name)) for name in names}
    reference_name = names[0]
    keys = sorted({key for result in values.values() for key in result})

    missing_keys = {
        name: sorted(set(keys) - set(result))
        for name, result in values.items()
        if set(result) != set(keys)
    }
    nonfinite_keys = {
        name: sorted(
            key
            for key, value in result.items()
            if not np.isfinite(float(value))
        )
        for name, result in values.items()
        if any(not np.isfinite(float(value)) for value in result.values())
    }

    errors: dict[str, dict[str, float]] = {}
    absolute_errors: dict[str, dict[str, float]] = {}
    worst_key = keys[0] if keys else ""
    worst_engine = reference_name
    worst_error = 0.0
    worst_absolute_error = 0.0
    for key in keys:
        per_engine: dict[str, float] = {}
        per_engine_absolute: dict[str, float] = {}
        baseline = values.get(reference_name, {}).get(key)
        for name in names:
            candidate = values.get(name, {}).get(key)
            if baseline is None or candidate is None:
                continue
            baseline_float = float(baseline)
            candidate_float = float(candidate)
            difference = abs(candidate_float - baseline_float)
            scale = max(abs(baseline_float), abs(candidate_float))
            error = difference / scale if scale > 0.0 else difference
            per_engine[name] = float(error)
            per_engine_absolute[name] = float(difference)
            if error > worst_error:
                worst_error, worst_key, worst_engine = float(error), key, name
            if difference > worst_absolute_error:
                worst_absolute_error = float(difference)
        errors[key] = per_engine
        absolute_errors[key] = per_engine_absolute

    values_agree = True
    for key in keys:
        baseline = values.get(reference_name, {}).get(key)
        if baseline is None:
            values_agree = False
            continue
        baseline_float = float(baseline)
        for name in names:
            candidate = values.get(name, {}).get(key)
            if candidate is None:
                values_agree = False
                continue
            candidate_float = float(candidate)
            scale = max(abs(baseline_float), abs(candidate_float))
            difference = abs(candidate_float - baseline_float)
            if difference > atol + rtol * scale:
                values_agree = False

    invalid_observables = bool(not keys or missing_keys or nonfinite_keys)
    return {
        "agree": bool(not invalid_observables and values_agree),
        "max_rel_error": worst_error,
        "max_absolute_error": worst_absolute_error,
        "worst_key": worst_key,
        "worst_engine": worst_engine,
        "reference": reference_name,
        "rtol": rtol,
        "atol": atol,
        "missing_keys": missing_keys,
        "nonfinite_keys": nonfinite_keys,
        "errors": errors,
        "absolute_errors": absolute_errors,
        "values": values,
    }


def format_report(report: Mapping) -> str:
    """One-screen summary of a comparison report."""
    lines = [
        f"reference engine : {report['reference']}",
        f"verdict          : {'AGREE' if report['agree'] else 'DISAGREE'} "
        f"(rtol {report['rtol']:g}, atol {report['atol']:g})",
        f"max rel error    : {report['max_rel_error']:.3e} "
        f"on {report['worst_key']} ({report['worst_engine']})",
    ]
    if report.get("missing_keys"):
        lines.append(f"missing keys     : {report['missing_keys']}")
    if report.get("nonfinite_keys"):
        lines.append(f"non-finite keys  : {report['nonfinite_keys']}")
    for key in sorted(report["values"][report["reference"]]):
        parts = [
            f"{name}={report['values'][name][key]:.12g}"
            for name in report["values"]
            if key in report["values"][name]
        ]
        lines.append(f"  {key:6s} " + "  ".join(parts))
    return "\n".join(lines)


def assert_engines_agree(
    runner: Callable[[str], Mapping[str, float]],
    engines: Iterable[str],
    rtol: float = 1e-6,
    atol: float = 0.0,
    name: str = "engines",
) -> dict:
    """Declare that independent engines agree; raise with the numbers if not.

    This is the CI-gate form of :func:`compare_across_engines`: independent
    implementations are a benchmark, and a benchmark that is never allowed to
    fail is not a benchmark.
    """
    report = compare_across_engines(runner, engines, rtol=rtol, atol=atol)
    if report["agree"]:
        return report
    raise ContractViolation(
        f"{name} disagree: {report['worst_key']} differs by "
        f"{report['max_rel_error']:.3e} between {report['reference']} and "
        f"{report['worst_engine']} (rtol {rtol:g}); "
        f"values "
        + ", ".join(
            f"{engine}={report['values'][engine].get(report['worst_key'])!r}"
            for engine in report["values"]
        )
        + (
            f"; missing keys {report['missing_keys']}"
            if report.get("missing_keys")
            else ""
        )
        + (
            f"; non-finite keys {report['nonfinite_keys']}"
            if report.get("nonfinite_keys")
            else ""
        )
    )


__all__ = ["compare_across_engines", "assert_engines_agree", "format_report", "np"]
