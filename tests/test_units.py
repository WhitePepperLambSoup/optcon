"""Unit and dimension algebra.

Design rule under test: SI treats the radian as dimensionless, but for
engineering safety this library keeps *angle* as its own base dimension, so
``mrad`` can never be silently added to a bare ratio.
"""

import pytest

from optcon import DimensionError, UnitError, unit


def test_millimetre_is_a_thousandth_of_a_metre():
    mm = unit("mm")
    assert mm.factor == pytest.approx(1e-3)
    assert mm.dimension == unit("m").dimension


def test_converting_mm_to_nm():
    assert unit("mm").to_value(1.0, unit("nm")) == pytest.approx(1e6)


def test_product_of_units_multiplies_factors_and_dims():
    mm2 = unit("mm") * unit("mm")
    assert mm2.factor == pytest.approx(1e-6)
    assert mm2.dimension == unit("m").dimension**2


def test_reciprocal_length_parses():
    inv_mm = unit("1/mm")
    assert inv_mm.factor == pytest.approx(1e3)
    assert inv_mm.dimension == unit("m").dimension**-1


def test_compound_unit_parses_products_and_quotients():
    mps = unit("m/s")
    assert mps.factor == pytest.approx(1.0)
    assert mps.dimension == (unit("m") / unit("s")).dimension


def test_prefixed_derived_units():
    thz = unit("THz")
    assert thz.factor == pytest.approx(1e12)
    assert thz.dimension == (unit("1/s")).dimension


def test_watt_is_derived_from_joule_per_second():
    assert unit("W").dimension == (unit("J") / unit("s")).dimension
    assert unit("mW").factor == pytest.approx(1e-3)


def test_percent_is_dimensionless_with_scale():
    assert unit("%").factor == pytest.approx(0.01)
    assert unit("%").dimension == unit("1").dimension


def test_angle_is_its_own_dimension():
    assert unit("rad").dimension != unit("1").dimension
    assert unit("deg").factor == pytest.approx(3.141592653589793 / 180.0)


def test_conversion_between_incompatible_dimensions_is_rejected():
    with pytest.raises(DimensionError):
        unit("mm").to_value(1.0, unit("mrad"))


def test_unknown_unit_name_is_rejected():
    with pytest.raises(UnitError):
        unit("furlong")


def test_unknown_prefix_is_rejected():
    with pytest.raises(UnitError):
        unit("qW")


def test_parenthesised_denominators_parse():
    radiance = unit("W/(m^2*sr)")
    assert radiance.factor == pytest.approx(1.0)
    assert radiance.dimension == (unit("W") / unit("m") ** 2 / unit("sr")).dimension


def test_a_parenthesised_group_behaves_like_the_starred_equivalent():
    assert unit("W/(m^2*sr*m)").dimension == unit("W/m^3/sr").dimension
    assert unit("W/(m^2*sr*m)").factor == pytest.approx(unit("W/m^3/sr").factor)


def test_nested_groups_and_exponents_compose():
    expression = unit("J/(m^2*s)")
    assert expression.dimension == unit("W/m^2").dimension
    assert expression.factor == pytest.approx(1.0)


def test_a_parenthesised_single_unit_is_allowed():
    assert unit("(m)").dimension == unit("m").dimension


def test_a_malformed_expression_is_rejected():
    for bad in ("W/(m^2", "W/", "W//m", "m^", "m^x", "()", "W/m)"):
        with pytest.raises(UnitError):
            unit(bad)
