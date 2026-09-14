"""Beam propagation engines, adjudicated by a closed-form reference.

Where Mie had two libraries to compare against each other, Gaussian beam
propagation has something better: an exact analytic answer.  w(z) =
w0*sqrt(1 + (z/zR)^2) is not an approximation, so an engine that disagrees
with it is simply wrong at that operating point.
"""

import numpy as np
import pytest

from optcon import q
from optcon.engines import (
    ENGINE_SPECS,
    available_engines,
    compare_across_engines,
    gaussian_beam_radius,
)

REFERENCE = "optcon_beam_reference"
FORVARD = "lightpipes_forvard"
FRESNEL = "lightpipes_fresnel"
BEAM_ENGINES = [name for name in (REFERENCE, FORVARD, FRESNEL) if name in available_engines()]

WAIST_MM = 1.0
WAVELENGTH_MM = 1.064e-3
RAYLEIGH_MM = np.pi * WAIST_MM**2 / WAVELENGTH_MM


def analytic_width(distance_mm: float) -> float:
    return WAIST_MM * np.sqrt(1.0 + (distance_mm / RAYLEIGH_MM) ** 2)


def test_beam_engines_are_registered():
    for name in (REFERENCE, FORVARD, FRESNEL):
        assert name in ENGINE_SPECS, name


@pytest.mark.skipif(REFERENCE not in BEAM_ENGINES, reason="reference unavailable")
def test_reference_reproduces_the_closed_form():
    result = gaussian_beam_radius(
        waist=q(WAIST_MM, "mm"),
        wavelength=q(WAVELENGTH_MM, "mm"),
        distance=q(RAYLEIGH_MM, "mm"),
        engine=REFERENCE,
    )
    assert result["w"].value == pytest.approx(analytic_width(RAYLEIGH_MM), rel=1e-12)


@pytest.mark.skipif(FORVARD not in BEAM_ENGINES, reason="lightpipes unavailable")
def test_spectral_propagation_matches_the_closed_form():
    result = gaussian_beam_radius(
        waist=q(WAIST_MM, "mm"),
        wavelength=q(WAVELENGTH_MM, "mm"),
        distance=q(RAYLEIGH_MM, "mm"),
        engine=FORVARD,
    )
    assert result["w"].value == pytest.approx(analytic_width(RAYLEIGH_MM), rel=1e-3)


@pytest.mark.skipif(FRESNEL not in BEAM_ENGINES, reason="lightpipes unavailable")
def test_convolution_propagation_carries_a_systematic_offset():
    """Characterisation of upstream behaviour, pinned as a regression guard."""
    result = gaussian_beam_radius(
        waist=q(WAIST_MM, "mm"),
        wavelength=q(WAVELENGTH_MM, "mm"),
        distance=q(RAYLEIGH_MM, "mm"),
        engine=FRESNEL,
    )
    deviation = result["w"].value / analytic_width(RAYLEIGH_MM) - 1.0
    assert 1e-2 < deviation < 2e-1  # a few percent wide, not a rounding artefact


@pytest.mark.skipif(FORVARD not in BEAM_ENGINES, reason="lightpipes unavailable")
def test_input_units_are_converted_at_the_boundary():
    in_mm = gaussian_beam_radius(
        waist=q(WAIST_MM, "mm"),
        wavelength=q(WAVELENGTH_MM, "mm"),
        distance=q(RAYLEIGH_MM, "mm"),
        engine=FORVARD,
    )
    in_mixed = gaussian_beam_radius(
        waist=q(1000.0, "um"),
        wavelength=q(1064.0, "nm"),
        distance=q(RAYLEIGH_MM / 1000.0, "m"),
        engine=FORVARD,
    )
    assert in_mm["w"].to_value("um") == pytest.approx(
        in_mixed["w"].to_value("um"), rel=1e-9
    )


@pytest.mark.skipif(len(BEAM_ENGINES) < 3, reason="needs all three beam engines")
def test_differential_comparison_separates_the_two_commands():
    def runner(engine):
        return {
            "w_um": gaussian_beam_radius(
                waist=q(WAIST_MM, "mm"),
                wavelength=q(WAVELENGTH_MM, "mm"),
                distance=q(RAYLEIGH_MM, "mm"),
                engine=engine,
            )["w"].to_value("um")
        }

    good = compare_across_engines(runner, [REFERENCE, FORVARD], rtol=1e-3)
    bad = compare_across_engines(runner, [REFERENCE, FRESNEL], rtol=1e-3)
    assert good["agree"] is True
    assert bad["agree"] is False
    assert bad["max_rel_error"] > 10 * good["max_rel_error"]


@pytest.mark.skipif("diffractio" not in available_engines("beam"), reason="diffractio unavailable")
def test_diffractio_matches_the_closed_form():
    """A chirp-z / Rayleigh-Sommerfeld propagator, a different method again."""
    result = gaussian_beam_radius(
        waist=q(50.0, "um"),
        wavelength=q(633.0, "nm"),
        distance=q(12.41, "mm"),
        grid_size=q(1.6, "mm"),
        samples=2048,
        engine="diffractio",
    )
    assert result["w"].to_value("um") == pytest.approx(70.71, rel=1e-2)


@pytest.mark.skipif("diffractio" not in available_engines("beam"), reason="diffractio unavailable")
def test_diffractio_refuses_a_grid_it_cannot_integrate_on():
    """Better a refusal than a number three orders of magnitude out."""
    with pytest.raises(ValueError, match="sampling"):
        gaussian_beam_radius(
            waist=q(WAIST_MM, "mm"),
            wavelength=q(WAVELENGTH_MM, "mm"),
            distance=q(RAYLEIGH_MM, "mm"),
            engine="diffractio",
        )


@pytest.mark.skipif("diffractio" not in available_engines("beam"), reason="diffractio unavailable")
def test_all_three_good_propagators_agree_with_each_other():
    """LightPipes spectral, diffractio CZT and optcon's own, three methods."""
    report = compare_across_engines(
        lambda engine: {
            "w_um": gaussian_beam_radius(
                waist=q(50.0, "um"),
                wavelength=q(633.0, "nm"),
                distance=q(12.41, "mm"),
                grid_size=q(1.6, "mm"),
                samples=2048,
                engine=engine,
            )["w"].to_value("um")
        },
        engines=[REFERENCE, FORVARD, "diffractio"],
        rtol=2e-2,
    )
    assert report["agree"] is True
