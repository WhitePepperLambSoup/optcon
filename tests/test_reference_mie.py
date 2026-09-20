"""An author-constructed Mie reference used to adjudicate the two engine libraries.

The reference is deliberately written from the textbook formula rather than
borrowed from either library, so that agreement (or disagreement) with it is
evidence about the libraries rather than about a shared code path.
"""

from typing import Any

import numpy as np
import pytest

from optcon import q
from optcon.engines import (
    available_engines,
    engine_spec,
    mie_efficiencies,
)
from optcon.engines.reference_mie import mie_efficiencies_reference

REFERENCE = "optcon_reference"


def rayleigh_limits(m: complex, x: float) -> tuple[float, float]:
    """Small-particle limits: Qsca = (8/3) x^4 |K|^2, Qext = 4 x Im(K)."""
    k = (m * m - 1.0) / (m * m + 2.0)
    return (8.0 / 3.0) * x**4 * abs(k) ** 2, 4.0 * x * k.imag


def size_parameter(diameter_nm: float, wavelength_nm: float) -> float:
    return np.pi * diameter_nm / wavelength_nm


def test_reference_matches_the_rayleigh_limit():
    m = 1.5 + 0.01j
    diameter_nm = 1.75106  # x = 0.01 at 550 nm
    x = size_parameter(diameter_nm, 550.0)
    expected_qsca, expected_qext = rayleigh_limits(m, x)

    result = mie_efficiencies_reference(m, diameter_nm, 550.0)

    assert result["Qsca"] == pytest.approx(expected_qsca, rel=1e-3)
    assert result["Qext"] == pytest.approx(expected_qext, rel=1e-3)


def test_reference_reports_no_absorption_for_a_lossless_sphere():
    result = mie_efficiencies_reference(1.5 + 0.0j, 200.0, 550.0)
    assert result["Qabs"] == pytest.approx(0.0, abs=1e-12)
    assert result["Qext"] == pytest.approx(result["Qsca"], rel=1e-12)


def test_reference_approaches_the_extinction_paradox_for_large_particles():
    result = mie_efficiencies_reference(1.5 + 0.0j, 50000.0, 550.0)
    assert result["Qext"] == pytest.approx(2.0, rel=5e-2)


def test_reference_is_registered_as_an_engine():
    spec = engine_spec(REFERENCE)
    assert spec.domain == "mie"
    assert spec.length_unit == "nm"
    assert REFERENCE in available_engines("mie")


@pytest.mark.skipif(
    not {"miepython", "PyMieScatt"} <= set(available_engines("mie")),
    reason="needs both Mie engines",
)
def test_reference_and_both_engines_agree_to_within_a_few_thousandths():
    kwargs: dict[str, Any] = dict(
        m=1.5 + 0.01j, diameter=q(200.0, "nm"), wavelength=q(550.0, "nm")
    )
    reference = mie_efficiencies_reference(1.5 + 0.01j, 200.0, 550.0)
    for engine in ("miepython", "PyMieScatt", REFERENCE):
        result = mie_efficiencies(engine=engine, **kwargs)
        assert result["Qext"].value == pytest.approx(reference["Qext"], rel=3e-3), engine


# ---------------------------------------------------------------------------
# Medium convention: the resolution of experiment 02's 0.2% mystery
# ---------------------------------------------------------------------------


def test_engines_agree_to_machine_precision_once_the_medium_is_explicit():
    """A stated vacuum medium removes any upstream default convention."""
    kwargs: dict[str, Any] = dict(
        m=1.5 + 0.01j,
        diameter=q(200.0, "nm"),
        wavelength=q(550.0, "nm"),
        medium_index=1.0,
    )
    reference = mie_efficiencies(engine=REFERENCE, **kwargs)["Qext"].value
    for engine in ("miepython", "PyMieScatt"):
        if engine not in available_engines("mie"):
            continue
        value = mie_efficiencies(engine=engine, **kwargs)["Qext"].value
        assert value == pytest.approx(reference, rel=1e-11), engine


def test_explicit_air_and_vacuum_queries_are_distinguishable():
    """The convention check uses stated media, not a version-specific default."""
    vacuum = mie_efficiencies_reference(1.5 + 0.01j, 200.0, 550.0, n_env=1.0)["Qext"]
    air = mie_efficiencies_reference(
        1.5 + 0.01j, 200.0, 550.0, n_env=1.00027316
    )["Qext"]
    assert 1e-4 < abs(air - vacuum) / vacuum < 1e-2


def test_a_real_medium_is_honoured_by_every_engine():
    """A refractive medium is where the wavelength conventions diverge."""
    kwargs: dict[str, Any] = dict(
        m=1.5 + 0.01j,
        diameter=q(200.0, "nm"),
        wavelength=q(550.0, "nm"),
        medium_index=1.5,
    )
    reference = mie_efficiencies(engine=REFERENCE, **kwargs)["Qext"].value
    for engine in ("miepython", "PyMieScatt"):
        if engine not in available_engines("mie"):
            continue
        value = mie_efficiencies(engine=engine, **kwargs)["Qext"].value
        assert value == pytest.approx(reference, rel=1e-9), engine
