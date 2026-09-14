"""Experiment 04 - a mature library, a closed-form answer, and a 7% gap.

The Mie experiment (03) needed a third implementation because two libraries
disagreeing says nothing about which one is wrong.  For Gaussian beam
propagation there is something better than a third opinion: the exact
relation w(z) = w0 sqrt(1 + (z/zR)^2).

LightPipes offers two propagators.  Run both against the closed form.

Run:  python -m optcon.examples.experiment_04_beam_adjudication
"""

from __future__ import annotations

import math

from optcon import q
from optcon.engines import available_engines, gaussian_beam_radius

WAIST_MM = 1.0
WAVELENGTH_MM = 1.064e-3
SAMPLES = 512
GRID_MM = 20.0
DISTANCES_IN_RAYLEIGH = (0.1, 0.5, 1.0, 2.0, 4.0)


def main() -> int:
    rayleigh = math.pi * WAIST_MM**2 / WAVELENGTH_MM
    engines = [
        name
        for name in ("optcon_beam_reference", "lightpipes_forvard", "lightpipes_fresnel")
        if name in available_engines("beam")
    ]
    reference_name = "optcon_beam_reference"
    if reference_name not in engines:
        print("the closed-form reference is unavailable; nothing to adjudicate against")
        return 1

    print("Gaussian beam, waist 1.00 mm, wavelength 1064 nm, 512 samples over 20 mm")
    print(f"Rayleigh range zR = {rayleigh:.3f} mm")
    print()
    header = (
        f"{'z/zR':>6s} {'closed form':>14s}  "
        + "  ".join(f"{name:>22s}" for name in engines)
        + "  " + "  ".join(f"{'dev ' + name:>22s}" for name in engines[1:])
    )
    print(header)
    print("-" * len(header))

    worst: dict[str, float] = {name: 0.0 for name in engines[1:]}
    for ratio in DISTANCES_IN_RAYLEIGH:
        distance_mm = ratio * rayleigh
        widths = {}
        for name in engines:
            widths[name] = gaussian_beam_radius(
                waist=q(WAIST_MM, "mm"),
                wavelength=q(WAVELENGTH_MM, "mm"),
                distance=q(distance_mm, "mm"),
                grid_size=q(GRID_MM, "mm"),
                samples=SAMPLES,
                engine=name,
            )["w"].value
        exact = widths[reference_name]
        deviations = []
        for name in engines[1:]:
            deviation = abs(widths[name] - exact) / exact
            worst[name] = max(worst[name], deviation)
            deviations.append(deviation)
        row = [f"{ratio:6.2f}", f"{exact:14.6f}"]
        row += [f"{widths[name]:22.6f}" for name in engines]
        row += [f"{deviation:22.3e}" for deviation in deviations]
        print("  ".join(row))

    print("\n=== verdict @ rtol 1e-3 ===")
    for name in engines[1:]:
        verdict = "agrees with the closed form" if worst[name] <= 1e-3 else "DISAGREES"
        print(f"{name:22s} worst deviation {worst[name]:.3e}   {verdict}")
    print(
        "\nBoth commands come from the same mature library and neither warns you.\n"
        "The convolution propagator is a few percent wide across the whole range,\n"
        "and the error does not shrink with finer sampling (256 to 4096 samples\n"
        "all land near +7%), so it is not a resolution problem. The spectral\n"
        "propagator reproduces the closed form to display precision."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
