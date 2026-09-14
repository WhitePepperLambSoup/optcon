"""The adoption layer: annotate a signature, get checked units.

Existing optical code is not going to be rewritten, so the only realistic
path is to let it keep its plain floats on the inside and declare units on
the signature.  That is what these decorators do, following the pattern that
prysm and poppy already proved works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, cast

import pytest

from optcon import (
    AMPLITUDE,
    POWER,
    AmplitudeOrderError,
    DimensionError,
    Quantity,
    power_ratio,
    q,
)
from optcon.specs import checked, to_quantities, units_of_dataclass


@checked
def free_space_raleigh_length(
    waist: Annotated[float, "um"], wavelength: Annotated[float, "nm"]
) -> Annotated[float, "um"]:
    """z_R = pi w0^2 / lambda, evaluated in the body's own units."""
    import math

    return math.pi * waist**2 / (wavelength * 1e-3)


@checked
def cavity_transmission(mirror: Annotated[float, "1", POWER]) -> Annotated[float, "1", POWER]:
    return 1.0 - mirror


@checked
def passthrough(value: Annotated[float, "mm"], extra):
    return value


def test_bare_numbers_are_interpreted_in_the_declared_unit():
    result = free_space_raleigh_length(50.0, 1064.0)
    assert isinstance(result, Quantity)
    assert result.unit.symbol == "um"
    assert result.value == pytest.approx(3.141592653589793 * 2500.0 / 1.064)


def test_a_compatible_quantity_is_converted_to_the_declared_unit():
    in_nm = free_space_raleigh_length(q(50.0, "um"), q(1064.0, "nm"))
    in_other = free_space_raleigh_length(q(0.05, "mm"), q(1.064, "um"))
    assert in_nm.value == pytest.approx(in_other.value, rel=1e-12)


def test_an_incompatible_quantity_is_rejected():
    with pytest.raises(DimensionError):
        free_space_raleigh_length(q(30.0, "deg"), q(1064.0, "nm"))


def test_return_annotation_types_the_result():
    result = cavity_transmission(0.02)
    assert result.unit.is_dimensionless
    assert result.amp_order == POWER


def test_a_returned_quantity_with_the_wrong_amplitude_order_is_rejected():
    @checked
    def wrong(watts: Annotated[float, "1", POWER]) -> Annotated[float, "1", AMPLITUDE]:
        # hands back a power where an amplitude was declared: the type
        # violation is the point of the test
        return power_ratio(watts)  # type: ignore[return-value]

    with pytest.raises(AmplitudeOrderError):
        wrong(0.5)


def test_a_returned_quantity_with_the_wrong_dimension_is_rejected():
    @checked
    def wrong(length: Annotated[float, "mm"]) -> Annotated[float, "s"]:
        # hands back a length where a time was declared
        return q(length, "mm")  # type: ignore[return-value]

    with pytest.raises(DimensionError):
        wrong(10.0)


def test_unannotated_return_stays_untyped_and_free_arguments_pass_through():
    # no return annotation was declared, so no unit is invented for the result
    assert passthrough(2.0, {"anything": 1}) == 2.0
    # the declared argument is still converted to millimetres
    assert passthrough(q(0.002, "m"), "kept") == pytest.approx(2.0)
    assert cast(Any, passthrough).__wrapped__(2.0, "kept") == 2.0


@dataclass
class MirrorOfResParams:
    """Field names copied from laser_sim_web/resonator.py's ResParams."""

    L1_mm: float = 300.0
    L2_mm: float = 300.0
    R1_mm: float = 1000.0
    wavelength_nm: float = 1064.0
    crystal_len_mm: float = 20.0
    pump_radius_um: float = 300.0
    tilt_mrad: float = 0.0
    beta: float = 1e-6


def _symbol(unit):
    """Test helper: the symbol of an inferred unit, asserting it was inferred."""
    assert unit is not None, "no unit was inferred from the field name"
    return unit.symbol


def test_units_are_inferred_from_field_name_suffixes():
    units = units_of_dataclass(MirrorOfResParams)
    assert _symbol(units["L1_mm"]) == "mm"
    assert _symbol(units["wavelength_nm"]) == "nm"
    assert _symbol(units["pump_radius_um"]) == "um"
    assert _symbol(units["tilt_mrad"]) == "mrad"


def test_fields_without_a_unit_suffix_are_left_untyped():
    assert units_of_dataclass(MirrorOfResParams)["beta"] is None


def test_a_dataclass_instance_converts_to_quantities():
    quantities = to_quantities(MirrorOfResParams())
    assert quantities["L1_mm"].to_value("m") == pytest.approx(0.3)
    assert quantities["wavelength_nm"].to_value("mm") == pytest.approx(1.064e-3)
    assert "beta" not in quantities


def test_checked_tuple_return_values():
    @checked
    def beam_splitter(
        power_in: Annotated[float, "W", POWER],
    ) -> tuple[Annotated[float, "W", POWER], Annotated[float, "W", POWER]]:
        half = power_in * 0.5
        return (half, half)

    t1, t2 = beam_splitter(q(2.0, "W", amp_order=POWER))
    assert isinstance(t1, Quantity)
    assert isinstance(t2, Quantity)
    assert t1.to_value("W") == pytest.approx(1.0)
    assert t2.to_value("mW") == pytest.approx(1000.0)
    assert t1.amp_order == POWER
    assert t2.amp_order == POWER

