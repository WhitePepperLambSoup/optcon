"""Integration with the optical engines already surveyed in this workspace.

The engines live in ``optical_simulation_projects/`` and
``ml_optical_projects/``; each one has its own unit convention.  The adapters
here are thin: they translate units and argument order at the boundary and
return typed quantities, so that crossing engines is checked rather than
remembered.
"""

import math
from typing import Any

import pytest

from optcon import ContractViolation, power_ratio, q
from optcon.engines import (
    ENGINE_SPECS,
    assert_engines_agree,
    available_engines,
    compare_across_engines,
    describe_engines,
    engine_spec,
    mie_efficiencies,
    stack_response,
)

THIN_FILM_ENGINES = [name for name in ("tmm_core", "tmm_fast") if name in available_engines()]
MIE_ENGINES = [name for name in ("miepython", "PyMieScatt") if name in available_engines()]
KNOWN_DOMAINS = {
    "thin_film",
    "mie",
    "beam",
    "diffraction",
    "ray_tracing",
    "lens_design",
    "photonic_circuit",
    "em_solver",
}


def test_registry_declares_a_unit_convention_per_engine():
    for name, spec in ENGINE_SPECS.items():
        assert spec.domain in KNOWN_DOMAINS, name
        # the whole point: every engine states the units it expects
        assert spec.length_unit, name
        assert spec.angle_unit, name


def test_registry_reports_availability():
    available = available_engines()
    assert isinstance(available, list)
    assert set(available) <= set(ENGINE_SPECS)
    # every available engine must also be described, and the closed-form
    # references ship with optcon itself so at least those always import
    assert "optcon_reference" in available or not available


def test_describe_engines_mentions_unit_conventions():
    text = describe_engines()
    assert "tmm" in text
    assert "nm" in text


def test_unknown_engine_is_rejected():
    with pytest.raises(KeyError):
        engine_spec("not_an_engine")


@pytest.mark.skipif(not THIN_FILM_ENGINES, reason="no thin-film engine importable")
def test_stack_response_returns_power_ratios():
    result = stack_response(
        n_list=[1.0, 2.0, 1.0],
        thicknesses=[math.inf, q(100.0, "nm"), math.inf],
        wavelength=q(550.0, "nm"),
        angle=q(0.0, "deg"),
        engine="tmm_core",
    )
    assert result["R"].amp_order == 2  # a power ratio, not an amplitude
    assert result["T"].amp_order == 2
    assert 0.0 <= result["R"].value <= 1.0
    assert result["R"].value + result["T"].value == pytest.approx(1.0, abs=1e-9)


@pytest.mark.skipif(not THIN_FILM_ENGINES, reason="no thin-film engine importable")
def test_wavelength_unit_is_respected_at_the_boundary():
    in_nm = stack_response(
        n_list=[1.0, 2.0, 1.0],
        thicknesses=[math.inf, q(100.0, "nm"), math.inf],
        wavelength=q(550.0, "nm"),
        angle=q(0.0, "deg"),
        engine="tmm_core",
    )
    in_um = stack_response(
        n_list=[1.0, 2.0, 1.0],
        thicknesses=[math.inf, q(0.1, "um"), math.inf],
        wavelength=q(0.55, "um"),
        angle=q(0.0, "deg"),
        engine="tmm_core",
    )
    assert in_nm["T"].value == pytest.approx(in_um["T"].value, rel=1e-12)


@pytest.mark.skipif(not THIN_FILM_ENGINES, reason="no thin-film engine importable")
def test_angle_unit_matters():
    normal = stack_response(
        n_list=[1.0, 2.0, 1.0],
        thicknesses=[math.inf, q(100.0, "nm"), math.inf],
        wavelength=q(550.0, "nm"),
        angle=q(30.0, "mrad"),
        engine="tmm_core",
    )
    tilted = stack_response(
        n_list=[1.0, 2.0, 1.0],
        thicknesses=[math.inf, q(100.0, "nm"), math.inf],
        wavelength=q(550.0, "nm"),
        angle=q(30.0, "deg"),
        engine="tmm_core",
    )
    assert normal["T"].value != pytest.approx(tilted["T"].value, rel=1e-6)


@pytest.mark.skipif(len(THIN_FILM_ENGINES) < 2, reason="needs two thin-film engines")
def test_two_thin_film_engines_agree_despite_different_internal_units():
    common: dict[str, Any] = dict(
        n_list=[1.0, 2.0, 1.0],
        thicknesses=[math.inf, q(100.0, "nm"), math.inf],
        wavelength=q(550.0, "nm"),
        angle=q(0.0, "deg"),
    )
    core = stack_response(engine="tmm_core", **common)
    fast = stack_response(engine="tmm_fast", **common)
    # two independent implementations: they agree to ~1e-9, not to machine
    # precision, because the summation order inside the stacks differs.
    assert core["T"].value == pytest.approx(fast["T"].value, rel=1e-7)
    assert core["R"].value == pytest.approx(fast["R"].value, rel=1e-7)


@pytest.mark.skipif(not MIE_ENGINES, reason="no Mie engine importable")
def test_mie_efficiencies_returns_dimensionless_ratios():
    result = mie_efficiencies(
        m=1.5 + 0.01j,
        diameter=q(200.0, "nm"),
        wavelength=q(550.0, "nm"),
        engine="miepython",
    )
    assert set(result) == {"Qext", "Qsca", "Qabs", "g"}
    assert result["Qabs"].is_dimensionless
    assert result["Qabs"].value == pytest.approx(
        result["Qext"].value - result["Qsca"].value, rel=1e-9
    )


@pytest.mark.skipif(not MIE_ENGINES, reason="no Mie engine importable")
def test_mie_diameter_and_wavelength_are_not_swappable():
    """Both engines take (diameter, wavelength) - but in opposite order."""
    straight = mie_efficiencies(
        m=1.5 + 0.01j,
        diameter=q(200.0, "nm"),
        wavelength=q(550.0, "nm"),
        engine="miepython",
    )
    swapped = mie_efficiencies(
        m=1.5 + 0.01j,
        diameter=q(550.0, "nm"),
        wavelength=q(200.0, "nm"),
        engine="miepython",
    )
    assert straight["Qext"].value != pytest.approx(swapped["Qext"].value, rel=1e-3)


@pytest.mark.skipif(len(MIE_ENGINES) < 2, reason="needs two Mie engines")
def test_compare_across_engines_reports_a_verdict():
    report = compare_across_engines(
        lambda engine: {
            key: quantity.value
            for key, quantity in mie_efficiencies(
                m=1.5 + 0.01j,
                diameter=q(200.0, "nm"),
                wavelength=q(550.0, "nm"),
                engine=engine,
            ).items()
        },
        engines=MIE_ENGINES,
        rtol=1e-9,
    )
    assert report["agree"] is True
    assert report["max_rel_error"] < 1e-9
    assert set(report["values"]) == set(MIE_ENGINES)


@pytest.mark.skipif(len(MIE_ENGINES) < 2, reason="needs two Mie engines")
def test_compare_across_engines_still_reports_a_real_disagreement():
    """The harness must not have been tuned into always saying yes.

    Feeding one engine a different physical question - here an explicitly
    stated air medium instead of vacuum - must still be caught.
    """

    def runner(engine):
        medium = 1.00027316 if engine == "PyMieScatt" else 1.0
        return {
            key: quantity.value
            for key, quantity in mie_efficiencies(
                m=1.5 + 0.01j,
                diameter=q(200.0, "nm"),
                wavelength=q(550.0, "nm"),
                medium_index=medium,
                engine=engine,
            ).items()
        }

    engines = [name for name in ("miepython", "PyMieScatt") if name in available_engines()]
    if len(engines) < 2:
        pytest.skip("needs both Mie engines")
    report = compare_across_engines(
        runner,
        engines=engines,
        rtol=1e-9,
    )
    assert report["agree"] is False
    assert report["max_rel_error"] > 1e-9
    assert report["worst_key"] in {"Qext", "Qsca", "Qabs", "g"}
    assert 1e-4 < report["max_rel_error"] < 1e-1


def test_compare_across_engines_rejects_missing_shared_observables():
    report = compare_across_engines(
        lambda engine: {"reflectance": 0.4}
        if engine == "reference"
        else {"transmittance": 0.6},
        engines=["reference", "candidate"],
    )

    assert report["agree"] is False
    assert report["missing_keys"] == {
        "reference": ["transmittance"],
        "candidate": ["reflectance"],
    }


def test_compare_across_engines_rejects_empty_observable_sets():
    report = compare_across_engines(
        lambda _engine: {},
        engines=["reference", "candidate"],
    )

    assert report["agree"] is False


def test_compare_across_engines_rejects_duplicate_engine_names():
    with pytest.raises(ValueError, match="unique engine names"):
        compare_across_engines(
            lambda _engine: {"value": 1.0},
            engines=["same", "same"],
        )


def test_compare_across_engines_rejects_nonfinite_observables():
    report = compare_across_engines(
        lambda engine: {"value": float("nan") if engine == "candidate" else 1.0},
        engines=["reference", "candidate"],
    )

    assert report["agree"] is False
    assert report["nonfinite_keys"] == {"candidate": ["value"]}


@pytest.mark.parametrize(
    ("reference", "candidate", "rtol", "atol"),
    [
        (1.0e12, 1.0e12 + 100.0, 1.0e-9, 0.0),
        (1.0e-12, 2.0e-12, 1.0e-3, 1.0e-9),
    ],
)
def test_compare_across_engines_applies_absolute_and_relative_tolerances_on_value_scale(
    reference, candidate, rtol, atol
):
    report = compare_across_engines(
        lambda engine: {"value": reference if engine == "reference" else candidate},
        engines=["reference", "candidate"],
        rtol=rtol,
        atol=atol,
    )

    assert report["agree"] is True


def test_power_ratio_helper_is_available_for_engine_outputs():
    assert power_ratio(0.9).amp_order == 2


@pytest.mark.skipif(len(MIE_ENGINES) < 2, reason="needs two Mie engines")
def test_assert_engines_agree_passes_when_tolerance_is_realistic():
    report = assert_engines_agree(
        lambda engine: {
            key: quantity.value
            for key, quantity in mie_efficiencies(
                m=1.5 + 0.01j,
                diameter=q(200.0, "nm"),
                wavelength=q(550.0, "nm"),
                engine=engine,
            ).items()
        },
        engines=MIE_ENGINES,
        rtol=1e-2,
    )
    assert report["agree"] is True


@pytest.mark.skipif(len(MIE_ENGINES) < 2, reason="needs two Mie engines")
def test_assert_engines_agree_raises_with_the_numbers():
    """The CI gate must fire when two engines are asked different questions."""

    def runner(engine):
        # deliberately different media: the gate should notice the 1e-3 offset
        medium = 1.0 if engine == MIE_ENGINES[0] else 1.001
        return {
            key: quantity.value
            for key, quantity in mie_efficiencies(
                m=1.5 + 0.01j,
                diameter=q(200.0, "nm"),
                wavelength=q(550.0, "nm"),
                medium_index=medium,
                engine=engine,
            ).items()
        }

    with pytest.raises(ContractViolation, match="disagree"):
        assert_engines_agree(
            runner, engines=MIE_ENGINES, rtol=1e-9, name="Mie efficiencies"
        )
