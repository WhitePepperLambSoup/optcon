"""End-to-end: design a Nd:YAG cavity and check every claim along the way.

This is the workflow the library exists for.  Nothing here is a special
case - the same six modules used in isolation in the test suite are used
together, and every number is either a closed-form relation or a contract
that either holds or raises.

Run:  python -m optcon.examples.design_a_laser
"""

from __future__ import annotations

import numpy as np

from optcon import ContractViolation, assert_passive, assert_unitary, q
from optcon.beam_quality import beam_parameter_product
from optcon.cavity import (
    finesse,
    free_spectral_range,
    linewidth,
    photon_lifetime,
    round_trip_gouy_phase,
    spot_size_at_mirrors,
    stability_product,
    transverse_mode_spacing,
    waist_size,
)
from optcon.elements import (
    compose,
    curved_mirror,
    effective_focal_length,
    free_space,
    thin_lens,
)
from optcon.gaussian import (
    beam_radius,
    propagate,
    q_from_waist,
    self_consistent_mode,
    waist_of,
)
from optcon.laser import (
    output_power,
    photon_lifetime_cavity,
    saturation_intensity,
    slope_efficiency,
    threshold_gain,
    threshold_pump_power,
)
from optcon.polarization import apply, half_wave_plate, intensity, jones_linear
from optcon.thinfilm import bragg_reflectance, single_layer_reflectance

WAVELENGTH = q(1064.0, "nm")
CAVITY_LENGTH = q(300.0, "mm")
MIRROR_RADIUS = q(500.0, "mm")
OUTPUT_COUPLER_REFLECTANCE = 0.95


def heading(text: str) -> None:
    print(f"\n--- {text} " + "-" * max(0, 58 - len(text)))


def main() -> int:
    print("Designing a two-mirror Nd:YAG cavity")
    print(f"  wavelength      {WAVELENGTH}")
    print(f"  cavity length   {CAVITY_LENGTH}")
    print(f"  mirror radius   {MIRROR_RADIUS}")

    heading("1. Is the resonator stable?")
    product = stability_product(CAVITY_LENGTH, MIRROR_RADIUS, MIRROR_RADIUS)
    g1, g2 = product**0.5, product**0.5
    print(f"  g1*g2 = {product:.4f}  (stable window is 0 <= g1*g2 < 1)")
    print(f"  round-trip Gouy phase = {round_trip_gouy_phase(g1, g2).to_value('rad'):.4f} rad")
    assert 0.0 <= product < 1.0, "the cavity must be stable"

    heading("2. What is the mode size, and does ABCD agree?")
    waist = waist_size(CAVITY_LENGTH, MIRROR_RADIUS, MIRROR_RADIUS, WAVELENGTH)
    on_mirror = spot_size_at_mirrors(CAVITY_LENGTH, MIRROR_RADIUS, MIRROR_RADIUS, WAVELENGTH)
    print(f"  waist           {waist.to_value('um'):8.2f} um")
    print(f"  spot on mirror  {on_mirror[0].to_value('um'):8.2f} um")

    round_trip = compose(
        free_space(CAVITY_LENGTH),
        curved_mirror(MIRROR_RADIUS),
        free_space(CAVITY_LENGTH),
        curved_mirror(MIRROR_RADIUS),
    )
    eigenmode = self_consistent_mode(round_trip, WAVELENGTH)
    abcd_radius = beam_radius(eigenmode, WAVELENGTH).to_value("um")
    print(f"  ABCD eigenmode  {abcd_radius:8.2f} um   (agreement to "
          f"{abs(abcd_radius - on_mirror[0].to_value('um')) / abcd_radius:.1e})")

    heading("3. Spectral properties")
    fsr = free_spectral_range(CAVITY_LENGTH)
    f = finesse(OUTPUT_COUPLER_REFLECTANCE)
    width = linewidth(fsr, f)
    print(f"  free spectral range {fsr.to_value('MHz'):8.3f} MHz")
    print(f"  finesse             {f:8.1f}")
    print(f"  linewidth           {width.to_value('MHz'):8.4f} MHz")
    print(f"  photon lifetime     {photon_lifetime(fsr, f).to_value('ns'):8.3f} ns")
    print(f"  transverse spacing  {transverse_mode_spacing(fsr, g1, g2).to_value('MHz'):8.3f} MHz")

    heading("4. Gain and pump requirements")
    gain = threshold_gain(CAVITY_LENGTH, output_coupler_reflectance=OUTPUT_COUPLER_REFLECTANCE)
    photon_life = photon_lifetime_cavity(
        CAVITY_LENGTH, output_coupler_reflectance=OUTPUT_COUPLER_REFLECTANCE
    )
    saturation = saturation_intensity(WAVELENGTH, cross_section=2.8e-23, lifetime=230e-6)
    mode_area = q(np.pi * (300.0**2), "um^2")
    threshold = threshold_pump_power(
        saturation, mode_area, round_trip_loss=-np.log(OUTPUT_COUPLER_REFLECTANCE)
    )
    print(f"  threshold gain      {gain.to_value('1/m'):8.3f} /m")
    print(f"  photon lifetime     {photon_life.to_value('ns'):8.3f} ns")
    print(f"  saturation intensity{saturation.to_value('kW/cm^2'):8.3f} kW/cm^2")
    print(f"  threshold pump      {threshold.to_value('W'):8.3f} W")

    efficiency = slope_efficiency(q(808.0, "nm"), WAVELENGTH, mode_overlap=0.85)
    for pump_watts in (0.5, 1.0, 2.0):
        produced = output_power(efficiency, pump_watts, threshold.to_value("W"))
        print(f"  {pump_watts:4.1f} W pump -> {produced * 1e3:7.1f} mW output")

    heading("5. Focusing the output")
    beam = q_from_waist(waist, WAVELENGTH)
    relay = compose(free_space(q(100.0, "mm")), thin_lens(q(75.0, "mm")))
    focused = propagate(beam, relay)
    print(f"  waists: {waist.to_value('um'):.2f} um -> "
          f"{waist_of(focused, WAVELENGTH).to_value('um'):.2f} um")
    print(f"  relay focal length  {effective_focal_length(relay).to_value('mm'):.2f} mm")
    print(f"  M^2 = 1 gives BPP    {beam_parameter_product(1.0, WAVELENGTH).to_value('mm*mrad'):.6f} mm mrad")

    heading("6. Optics that must satisfy a contract")
    plate = half_wave_plate(q(45.0, "deg"))
    assert_unitary(plate.value, name="half-wave plate")
    rotated = apply(plate, jones_linear(q(0.0, "deg")))
    print(f"  HWP at 45 deg is unitary; transmitted fraction "
          f"{intensity(rotated).to_value('1'):.4f}")

    for reflectance_value in (0.95, 0.99, 0.999):
        assert_passive(np.array([[reflectance_value**0.5]]), name="mirror")
    print("  mirrors declared passive: ok")

    try:
        assert_passive(np.array([[1.4]]), name="amplifying mirror")
    except ContractViolation as error:
        print(f"  a mirror with gain 1.4 is rejected: {type(error).__name__}")

    heading("7. Coatings")
    anti_reflection = single_layer_reflectance(
        1.0, 1.38, 1.5, q(192.8, "nm"), WAVELENGTH
    )
    bragg = bragg_reflectance(1.0, 2.3, 1.45, 1.5, pairs=20)
    print(f"  MgF2 single layer on glass: R = {anti_reflection.to_value('1') * 100:.2f}%")
    print(f"  20-pair quarter-wave stack: R = {bragg * 100:.4f}%")

    print("\nEvery number above is a closed form or a contract; none was fitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
