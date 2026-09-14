"""Cross-engine differential testing.

Independent implementations of the same physics are the cheapest source of
ground truth a computational field has, and almost nobody uses them.  This
module runs the same physical query through several engines, normalises the
outputs through optcon's unit layer, and reports whether they agree.
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

    values = {name: dict(runner(name)) for name in names}
    reference_name = names[0]
    keys = sorted({key for result in values.values() for key in result})

    errors: dict[str, dict[str, float]] = {}
    worst_key = keys[0] if keys else ""
    worst_engine = reference_name
    worst_error = 0.0
    for key in keys:
        per_engine: dict[str, float] = {}
        baseline = values.get(reference_name, {}).get(key)
        for name in names:
            candidate = values.get(name, {}).get(key)
            if baseline is None or candidate is None:
                continue
            scale = max(abs(float(baseline)), abs(float(candidate)))
            difference = abs(float(candidate) - float(baseline))
            error = difference / scale if scale > 0 else difference
            per_engine[name] = float(error)
            if error > worst_error:
                worst_error, worst_key, worst_engine = float(error), key, name
        errors[key] = per_engine

    tolerance = rtol + atol
    return {
        "agree": bool(worst_error <= tolerance),
        "max_rel_error": worst_error,
        "worst_key": worst_key,
        "worst_engine": worst_engine,
        "reference": reference_name,
        "rtol": rtol,
        "errors": errors,
        "values": values,
    }


def format_report(report: Mapping) -> str:
    """One-screen summary of a comparison report."""
    lines = [
        f"reference engine : {report['reference']}",
        f"verdict          : {'AGREE' if report['agree'] else 'DISAGREE'} "
        f"(rtol {report['rtol']:g})",
        f"max rel error    : {report['max_rel_error']:.3e} "
        f"on {report['worst_key']} ({report['worst_engine']})",
    ]
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
        f"values " + ", ".join(
            f"{engine}={report['values'][engine].get(report['worst_key'])!r}"
            for engine in report["values"]
        )
    )


__all__ = ["compare_across_engines", "assert_engines_agree", "format_report", "np"]
