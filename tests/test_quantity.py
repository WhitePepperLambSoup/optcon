"""Quantity arithmetic and the amplitude/power distinction.

This is the layer that has no counterpart in pint/unyt: a field amplitude and
a power ratio are different objects, and ``sqrt`` on a power ratio must yield
an amplitude ratio while ``sqrt`` on an amplitude ratio is a bug.
"""

import numpy as np
import pytest

from optcon import (
    AmplitudeOrderError,
    DimensionError,
    amplitude_ratio,
    dimensionless,
    power_ratio,
    q,
    sqrt,
)


def test_converting_quantity_to_another_unit():
    assert q(300.0, "mm").to_value("m") == pytest.approx(0.3)


def test_adding_compatible_units_converts_to_left_operand():
    total = q(1.0, "mm") + q(1.0, "um")
    assert total.unit.symbol == "mm"
    assert total.value == pytest.approx(1.001)


def test_adding_incompatible_dimensions_is_rejected():
    with pytest.raises(DimensionError):
        q(1.0, "mm") + q(1.0, "mrad")


def test_multiplying_quantities_composes_units_and_scales():
    area = q(2.0, "mm") * q(3.0, "mm")
    assert area.to_value("mm^2") == pytest.approx(6.0)
    assert area.to_value("m^2") == pytest.approx(6e-6)


def test_dividing_quantities_composes_units():
    rate = dimensionless(1.0) / q(2.0, "ms")
    assert rate.to_value("1/s") == pytest.approx(500.0)


def test_power_of_quantity_scales_units():
    squared = q(2.0, "mm") ** 2
    assert squared.value == pytest.approx(4.0)
    assert squared.to_value("m^2") == pytest.approx(4e-6)


def test_sqrt_of_power_ratio_yields_amplitude_ratio():
    field = sqrt(power_ratio(0.81))
    assert field.value == pytest.approx(0.9)
    assert field.amp_order == 1


def test_sqrt_of_amplitude_ratio_is_rejected():
    with pytest.raises(AmplitudeOrderError):
        sqrt(amplitude_ratio(0.5))


def test_adding_amplitude_and_power_is_rejected():
    with pytest.raises(AmplitudeOrderError):
        amplitude_ratio(0.5) + power_ratio(0.5)


def test_bare_float_is_treated_as_same_unit_and_untracked_order():
    total = q(2.0, "mm") + 3.0
    assert total.unit.symbol == "mm"
    assert total.value == pytest.approx(5.0)


def test_multiplying_amplitude_by_amplitude_gives_power():
    product = amplitude_ratio(0.9) * amplitude_ratio(0.9)
    assert product.amp_order == 2
    assert product.value == pytest.approx(0.81)


def test_order_is_preserved_when_scaling_by_a_plain_ratio():
    scaled = amplitude_ratio(2.0) * dimensionless(0.5)
    assert scaled.amp_order == 1
    assert scaled.value == pytest.approx(1.0)


def test_exp_requires_dimensionless_argument():
    with pytest.raises(DimensionError):
        np.exp(q(1.0, "mm"))


def test_sine_requires_an_angle():
    with pytest.raises(DimensionError):
        np.sin(dimensionless(1.0))


def test_sine_of_degrees_is_evaluated_correctly():
    result = np.sin(q(90.0, "deg"))
    assert result.to_value("1") == pytest.approx(1.0)


def test_arctangent_returns_an_angle():
    angle = np.arctan(q(1.0, "1"))
    assert angle.to_value("rad") == pytest.approx(np.pi / 4)


def test_array_valued_quantities_multiply_elementwise():
    scaled = q(np.array([1.0, 2.0]), "mm") * dimensionless(2.0)
    assert list(scaled.to_value("mm")) == pytest.approx([2.0, 4.0])


def test_reduction_preserves_unit():
    total = q(np.array([1.0, 2.0, 3.0]), "mm").sum()
    assert total.to_value("mm") == pytest.approx(6.0)


def test_explicit_unit_mismatch_in_comparison_is_rejected():
    with pytest.raises(DimensionError):
        # numpy's ufunc stub does not know about duck typing; the point of the
        # test is that the mismatch is caught at run time
        bool(np.greater(q(1.0, "mm"), q(1.0, "s")))  # type: ignore[call-overload]


def test_a_plain_number_can_be_raised_to_a_quantity_exponent():
    result = np.power(2.0, dimensionless(3.0))  # type: ignore[call-overload]
    assert result.is_dimensionless
    assert result.value == pytest.approx(8.0)


def test_a_dimensioned_exponent_is_rejected():
    with pytest.raises(DimensionError):
        2.0 ** q(3.0, "mm")


def test_quantity_comparisons_and_equality():
    q1 = q(2.0, "mm")
    q2 = q(2000.0, "um")
    q3 = q(3.0, "mm")

    assert q1 == q2
    assert q1 != q3
    assert q1 < q3
    assert q1 <= q2
    assert q3 > q1
    assert q3 >= q2

    # Compare with bare number
    assert q1 == 2.0
    assert bool(q1 != 3.0) is True

    # Incompatible dimension equality returns False
    assert not (q(1.0, "mm") == q(1.0, "s"))
    assert (q(1.0, "mm") != q(1.0, "s"))


def test_reverse_arithmetic_operations():
    # __rsub__: 5.0 - 2 mm -> 3 mm
    sub_res = 5.0 - q(2.0, "mm")
    assert sub_res.value == pytest.approx(3.0)
    assert sub_res.unit.symbol == "mm"

    # __rtruediv__: 10 / 2 mm -> 5 1/mm
    div_res = 10.0 / q(2.0, "mm")
    assert div_res.to_value("1/m") == pytest.approx(5000.0)

    # __rpow__ with amplitude order error
    with pytest.raises(AmplitudeOrderError):
        2.0 ** amplitude_ratio(2)


def test_quantity_attributes_and_conversions():
    x = q(500.0, "nm", amp_order=1)
    assert x.magnitude == 500.0
    assert x.si_value == pytest.approx(5e-7)
    assert not x.is_dimensionless
    assert x.has_same_dimension_as(q(1.0, "m"))
    assert not x.has_same_dimension_as(q(1.0, "s"))

    # to() and to_value() with Unit instance
    from optcon.units import unit as _unit
    u_m = _unit("m")
    assert x.to(u_m).value == pytest.approx(5e-7)
    assert x.to_value(u_m) == pytest.approx(5e-7)


def test_numpy_ufunc_branch_coverage():
    from optcon.errors import UnitError

    q1 = q(2.0, "mm")
    q2 = q(3.0, "mm")

    # maximum / minimum
    max_q = np.maximum(q1, q2)  # type: ignore[call-overload]
    assert max_q.value == pytest.approx(3.0)
    min_q = np.minimum(q1, q2)  # type: ignore[call-overload]
    assert min_q.value == pytest.approx(2.0)

    # square, reciprocal
    assert np.square(q1).to_value("mm^2") == pytest.approx(4.0)
    assert np.reciprocal(q1).to_value("1/mm") == pytest.approx(0.5)

    # reductions
    arr = q(np.array([2.0, 3.0, 4.0]), "mm")
    assert np.add.reduce(arr).value == pytest.approx(9.0)  # type: ignore[call-overload]

    # in-place ufunc is rejected
    with pytest.raises(UnitError):
        np.add(q1, q2, out=np.zeros(1))  # type: ignore[call-overload]

    # array exponent rejected
    with pytest.raises(UnitError):
        q1 ** np.array([1, 2])

