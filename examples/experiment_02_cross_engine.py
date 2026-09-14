"""Experiment 02 - how much do independent optical engines disagree?

The workspace ships two thin-film engines and two Mie engines.  Nobody
normally runs them against each other, which means nobody knows the answer to
the question this experiment asks.

All geometry crosses the optcon boundary as typed quantities, so each engine
receives its own native units without the caller tracking them.

Run:  python -m optcon.examples.experiment_02_cross_engine
"""

from __future__ import annotations

import math

from optcon import q
from optcon.engines import (
    available_engines,
    compare_across_engines,
    describe_engines,
    mie_efficiencies,
    stack_response,
)

WAVELENGTHS_NM = (450.0, 550.0, 650.0)
ANGLES_DEG = (0.0, 30.0)
DIAMETERS_NM = (50.0, 200.0, 1000.0, 5000.0)
MOLAR_M = 1.5 + 0.01j


def _cross_engine_tolerance() -> float:
    """How closely should two independent implementations agree?"""
    return 1e-6


def thin_film_table(engines, tolerance):
    print("\n=== thin film: tmm_core vs tmm_fast ===")
    header = (
        f"{'wavelength':>10s} {'angle':>7s}  "
        + "  ".join(f"{name + ' T':>16s}" for name in engines)
        + f"  {'rel err':>10s}  verdict"
    )
    print(header)
    print("-" * len(header))

    worst = 0.0
    for wavelength_nm in WAVELENGTHS_NM:
        for angle_deg in ANGLES_DEG:
            def run(engine: str, w: float = wavelength_nm, a: float = angle_deg):
                return {
                    key: value.value
                    for key, value in stack_response(
                        n_list=[1.0, 2.0, 1.0],
                        thicknesses=[math.inf, q(100.0, "nm"), math.inf],
                        wavelength=q(w, "nm"),
                        angle=q(a, "deg"),
                        engine=engine,
                    ).items()
                }

            report = compare_across_engines(
                run,
                engines=engines,
                rtol=tolerance,
            )
            worst = max(worst, report["max_rel_error"])
            transmittances = "  ".join(
                f"{report['values'][name]['T']:16.12f}" for name in engines
            )
            verdict = "agree" if report["agree"] else "DISAGREE"
            print(
                f"{wavelength_nm:8.0f}nm {angle_deg:5.0f}deg  {transmittances}  "
                f"{report['max_rel_error']:10.2e}  {verdict}"
            )
    return worst


def mie_table(engines, tolerance):
    print("\n=== Mie scattering: miepython vs PyMieScatt ===")
    header = (
        f"{'diameter':>10s} {'size x':>8s}  "
        + "  ".join(f"{name + ' Qext':>16s}" for name in engines)
        + f"  {'rel err':>10s}  verdict"
    )
    print(header)
    print("-" * len(header))

    worst = 0.0
    for diameter_nm in DIAMETERS_NM:
        def run_mie(engine: str, d: float = diameter_nm):
            return {
                key: value.value
                for key, value in mie_efficiencies(
                    m=MOLAR_M,
                    diameter=q(d, "nm"),
                    wavelength=q(550.0, "nm"),
                    engine=engine,
                ).items()
            }

        report = compare_across_engines(
            run_mie,
            engines=engines,
            rtol=tolerance,
        )
        worst = max(worst, report["max_rel_error"])
        extinctions = "  ".join(
            f"{report['values'][name]['Qext']:16.10f}" for name in engines
        )
        verdict = "agree" if report["agree"] else "DISAGREE"
        print(
            f"{diameter_nm:8.0f}nm {math.pi * diameter_nm / 550.0:8.3f}  "
            f"{extinctions}  {report['max_rel_error']:10.2e}  {verdict}"
        )
        print(f"{'':21s}worst key: {report['worst_key']}")
    return worst


def main() -> int:
    print("=== registered engines and the units they insist on ===")
    print(describe_engines())

    tolerance = _cross_engine_tolerance()
    print(f"\nagreement tolerance: rtol = {tolerance:g}")

    thin_film_engines = [
        name for name in ("tmm_core", "tmm_fast") if name in available_engines()
    ]
    mie_engines = [
        name for name in ("miepython", "PyMieScatt") if name in available_engines()
    ]

    thin_worst = None
    if len(thin_film_engines) == 2:
        thin_worst = thin_film_table(thin_film_engines, tolerance)

    mie_worst = None
    if len(mie_engines) == 2:
        mie_worst = mie_table(mie_engines, tolerance)

    print("\n=== summary ===")
    if thin_worst is not None:
        print(f"thin film  worst relative disagreement : {thin_worst:.3e}")
    if mie_worst is not None:
        print(f"Mie        worst relative disagreement : {mie_worst:.3e}")
    print(
        "\nTwo independent implementations of the same physics do not have to\n"
        "agree, and nothing in either library tells you that. Declaring a\n"
        "tolerance turns that from an unknown into a claim you can check."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
