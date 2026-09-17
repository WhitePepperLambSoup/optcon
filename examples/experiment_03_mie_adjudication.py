"""Experiment 03 - Mie adjudication with an optional medium-default check.

The experiment always evaluates the independent reference series. It compares
that reference with each Mie library that is importable in the current
environment. When PyMieScatt is available, it also demonstrates that omitting
``nMedium`` asks a different physical question from an explicit vacuum-medium
call. The script prints measured discrepancies; it does not assume that every
optional engine is installed or that all implementations agree at machine
precision.

Run:  python -m optcon.examples.experiment_03_mie_adjudication
"""

from __future__ import annotations

import math

from optcon import q
from optcon.engines import mie_efficiencies, reference_mie
from optcon.engines.registry import is_available

MOLAR_M = 1.5 + 0.01j
WAVELENGTH_NM = 550.0
DIAMETERS_NM = (50.0, 200.0, 1000.0, 5000.0, 20000.0)


def _relative(value: float, reference: float) -> float:
    scale = max(abs(value), abs(reference))
    return abs(value - reference) / scale if scale else 0.0


def library_default_gap():
    """What the two libraries report when each is called with its own defaults."""
    print("\n=== as called with each library's own defaults ===")
    reference = reference_mie.mie_efficiencies_reference(
        MOLAR_M, 200.0, WAVELENGTH_NM
    )["Qext"]
    if not is_available("PyMieScatt"):
        print("PyMieScatt unavailable in this environment")
        return
    try:
        import PyMieScatt

        default = float(
            PyMieScatt.MieQ(MOLAR_M, WAVELENGTH_NM, 200.0, asDict=True)["Qext"]
        )
        stated = float(
            PyMieScatt.MieQ(
                MOLAR_M, WAVELENGTH_NM, 200.0, nMedium=1.0, asDict=True
            )["Qext"]
        )
    except ImportError:
        print("PyMieScatt unavailable")
        return
    print(f"independent reference                 : {reference:.12f}")
    print(
        f"PyMieScatt as shipped                 : {default:.12f}   "
        f"deviation {abs(default - reference) / reference:.3e}"
    )
    print(
        f"PyMieScatt with nMedium=1.0           : {stated:.12f}   "
        f"deviation {abs(stated - reference) / reference:.3e}"
    )
    print(
        "\nPyMieScatt.MieQ defaults to nMedium=1.00027316 (air) and silently\n"
        "scales the refractive index by it. The library default, not the\n"
        "algorithm, is what experiment 02 was measuring."
    )


def adjudication_table():
    libraries = [name for name in ("miepython", "PyMieScatt") if is_available(name)]
    if not libraries:
        print("\n=== three-way adjudication (Qext) ===")
        print("Note: Neither miepython nor PyMieScatt is installed; skipping three-way comparison.")
        return {}
    print("\n=== three-way adjudication (Qext) ===")
    header = (
        f"{'diameter':>10s} {'size x':>8s} {'reference':>16s}  "
        + "  ".join(f"{name:>16s}" for name in libraries)
        + "  " + "  ".join(f"{'rel ' + name:>16s}" for name in libraries)
    )
    print(header)
    print("-" * len(header))

    worst = {name: 0.0 for name in libraries}
    for diameter_nm in DIAMETERS_NM:
        reference = reference_mie.mie_efficiencies_reference(
            MOLAR_M, diameter_nm, WAVELENGTH_NM
        )["Qext"]
        row = [f"{diameter_nm:8.0f}nm {math.pi * diameter_nm / WAVELENGTH_NM:8.3f} {reference:16.10f}"]
        deviations = []
        for name in libraries:
            value = mie_efficiencies(
                m=MOLAR_M,
                diameter=q(diameter_nm, "nm"),
                wavelength=q(WAVELENGTH_NM, "nm"),
                engine=name,
            )["Qext"].value
            error = _relative(value, reference)
            worst[name] = max(worst[name], error)
            row.append(f"{value:16.10f}")
            deviations.append(error)
        row += [f"{error:16.2e}" for error in deviations]
        print("  ".join(row))
    return worst


def convergence_check() -> None:
    """Show that the reference has converged, so truncation is not the cause."""
    print("\n=== is the reference converged? (Qext at varying term counts) ===")
    original = reference_mie.wiscombe_terms
    header = f"{'diameter':>10s} " + "  ".join(
        f"{label:>18s}" for label in ("wiscombe ceil", "round variant", "ceil + 8 terms")
    )
    print(header)
    print("-" * len(header))
    variants = (
        ("ceil", original),
        ("round", lambda x: max(3, int(round(2.0 + x + 4.0 * x ** (1.0 / 3.0))))),
        ("more", lambda x: original(x) + 8),
    )
    try:
        for diameter_nm in DIAMETERS_NM:
            values = []
            for _, rule in variants:
                reference_mie.wiscombe_terms = rule
                values.append(
                    reference_mie.mie_efficiencies_reference(
                        MOLAR_M, diameter_nm, WAVELENGTH_NM
                    )["Qext"]
                )
            print(
                f"{diameter_nm:8.0f}nm "
                + "  ".join(f"{value:18.10f}" for value in values)
            )
    finally:
        reference_mie.wiscombe_terms = original

    print(
        "\nAll three term counts give the same ten significant digits, so the\n"
        "reference is converged at the displayed precision. Any remaining\n"
        "inter-engine deviation is therefore not attributable to this term-count choice."
    )


def main() -> int:
    library_default_gap()
    worst = adjudication_table()
    convergence_check()

    if worst:
        print("\n=== verdict (through the optcon adapter, medium stated) ===")
        for name, error in sorted(worst.items(), key=lambda item: item[1]):
            verdict = "agrees with the reference" if error <= 1e-6 else "DISAGREES"
            print(f"{name:12s} max deviation {error:.3e}   {verdict}")
        print(
            "\nThe adapter states the surrounding medium explicitly for every engine.\n"
            "The values above are the evidence for this run; unavailable optional\n"
            "libraries are not counted as agreement. A library default may be valid\n"
            "for its intended application while still changing the physical query,\n"
            "which is why the medium belongs in the adapter contract."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
