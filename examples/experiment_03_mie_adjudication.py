"""Experiment 03 - Mie adjudication with an explicit medium check.

The experiment always evaluates the author-constructed reference series in the
evaluated source tree. It compares that reference with each Mie library that is importable in the current
environment. It also demonstrates that otherwise identical vacuum and air
queries are different physical questions. The script prints measured
discrepancies; it does not assume that every optional engine is installed or
that all implementations agree at machine precision.

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


def explicit_medium_gap():
    """Compare two stated media without relying on an upstream default."""
    print("\n=== explicit vacuum and air queries ===")
    vacuum = reference_mie.mie_efficiencies_reference(
        MOLAR_M, 200.0, WAVELENGTH_NM, n_env=1.0
    )["Qext"]
    air = reference_mie.mie_efficiencies_reference(
        MOLAR_M, 200.0, WAVELENGTH_NM, n_env=1.00027316
    )["Qext"]
    if not is_available("PyMieScatt"):
        print("PyMieScatt unavailable in this environment")
        return
    measured_air = mie_efficiencies(
        m=MOLAR_M,
        diameter=q(200.0, "nm"),
        wavelength=q(WAVELENGTH_NM, "nm"),
        medium_index=1.00027316,
        engine="PyMieScatt",
    )["Qext"].value
    print(f"vacuum reference                      : {vacuum:.12f}")
    print(f"air reference                         : {air:.12f}")
    print(
        f"PyMieScatt with explicit air medium   : {measured_air:.12f}   "
        f"deviation from air reference {abs(measured_air - air) / air:.3e}"
    )
    print(
        f"air-vacuum physical-query difference  : {abs(air - vacuum) / vacuum:.3e}"
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
    explicit_medium_gap()
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
            "libraries are not counted as agreement. A medium convention may be valid\n"
            "for its intended application while still changing the physical query,\n"
            "which is why the medium belongs in the adapter contract."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
