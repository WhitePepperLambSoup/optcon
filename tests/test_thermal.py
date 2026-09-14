"""Thermal lensing and Gaussian aperture truncation.

Both are end-pumped-laser workhorses: the thermal lens sets how much pump
power a cavity can take before its mode changes, and the aperture loss sets
how tight the mode can be before the mirrors start clipping it.
"""

import numpy as np
import pytest

from optcon import DimensionError, q
from optcon.thermal import (
    gaussian_aperture_transmission,
    gaussian_clipping_loss,
    thermal_lens_focal_length,
)

PUMP_RADIUS_UM = 300.0
PUMP_POWER_W = 10.0


def test_thermal_focal_length_follows_the_standard_formula():
    value = thermal_lens_focal_length(
        pump_power=q(PUMP_POWER_W, "W"),
        pump_radius=q(PUMP_RADIUS_UM, "um"),
        thermal_conductivity=1.4,
        dn_dt=7.3e-6,
        absorbed_fraction=0.8,
    )
    radius_m = PUMP_RADIUS_UM * 1e-6
    absorbed = PUMP_POWER_W * 0.8
    expected = 2.0 * np.pi * 1.4 * radius_m**2 / (absorbed * 7.3e-6)
    assert value.to_value("mm") == pytest.approx(expected * 1e3)


def test_a_stronger_pump_shortens_the_thermal_focal_length():
    weak = thermal_lens_focal_length(
        q(5.0, "W"), q(PUMP_RADIUS_UM, "um"), 1.4, 7.3e-6, 0.8
    ).to_value("mm")
    strong = thermal_lens_focal_length(
        q(10.0, "W"), q(PUMP_RADIUS_UM, "um"), 1.4, 7.3e-6, 0.8
    ).to_value("mm")
    assert strong == pytest.approx(weak / 2.0)


def test_thermal_focal_length_scales_with_the_square_of_the_pump_radius():
    small = thermal_lens_focal_length(
        q(PUMP_POWER_W, "W"), q(200.0, "um"), 1.4, 7.3e-6, 0.8
    ).to_value("mm")
    large = thermal_lens_focal_length(
        q(PUMP_POWER_W, "W"), q(400.0, "um"), 1.4, 7.3e-6, 0.8
    ).to_value("mm")
    assert large == pytest.approx(4.0 * small)


def test_a_negative_dn_dt_needs_an_explicit_convention():
    with pytest.raises(ValueError, match="dn_dt"):
        thermal_lens_focal_length(q(PUMP_POWER_W, "W"), q(PUMP_RADIUS_UM, "um"), 1.4, 0.0, 0.8)


def test_gaussian_aperture_transmission_and_clipping_are_complementary():
    for ratio in (0.5, 1.0, 1.5, 2.0):
        aperture = q(ratio * 500.0, "um")
        waist = q(500.0, "um")
        transmitted = gaussian_aperture_transmission(aperture, waist)
        clipped = gaussian_clipping_loss(aperture, waist)
        assert transmitted + clipped == pytest.approx(1.0)
        assert transmitted == pytest.approx(1.0 - np.exp(-2.0 * ratio**2))


def test_a_large_aperture_clips_almost_nothing():
    assert gaussian_clipping_loss(q(2500.0, "um"), q(500.0, "um")).to_value("1") == pytest.approx(
        np.exp(-50.0), abs=1e-12
    )


def test_an_aperture_equal_to_the_waist_passes_about_86_percent():
    assert gaussian_aperture_transmission(q(500.0, "um"), q(500.0, "um")).to_value(
        "1"
    ) == pytest.approx(0.8647, rel=1e-4)


def test_clipping_loss_is_a_power_ratio():
    assert gaussian_clipping_loss(q(300.0, "um"), q(500.0, "um")).amp_order == 2


def test_a_dimensioned_pump_power_is_rejected():
    with pytest.raises((DimensionError, ValueError)):
        thermal_lens_focal_length(
            q(10.0, "mm"), q(PUMP_RADIUS_UM, "um"), 1.4, 7.3e-6, 0.8
        )


def test_an_absorbed_fraction_above_one_is_rejected():
    with pytest.raises(ValueError, match="absorbed_fraction"):
        thermal_lens_focal_length(q(10.0, "W"), q(PUMP_RADIUS_UM, "um"), 1.4, 7.3e-6, 1.5)
