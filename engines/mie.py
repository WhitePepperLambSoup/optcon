"""Typed Mie efficiencies on top of the surveyed Mie engines.

Three conventions have to be reconciled here, and every one of them is a
silent trap when it is not:

1. **Argument order** - miepython takes ``(diameter, wavelength)``,
   PyMieScatt takes ``(wavelength, diameter)``.  Keyword-only arguments make
   the swap impossible.
2. **Absorption** - miepython returns ``(qext, qsca, qback, g)``; absorption
   is not in its return value at all and has to come out of the extinction
   budget.  PyMieScatt reports it directly.
3. **The surrounding medium** - released PyMieScatt artifacts disagree about
   whether ``MieQ`` converts the wavelength when ``nMedium`` is supplied.
   This adapter avoids that unstable boundary: it converts the refractive
   index and vacuum wavelength to their in-medium values itself, then calls
   PyMieScatt with ``nMedium=1``.
"""

from __future__ import annotations

from typing import Any

from ..quantity import RATIO, Quantity
from ..units import unit as _unit
from .registry import require_available

_KEYS = ("Qext", "Qsca", "Qabs", "g")


def _length_in(value: Any, target: str, field: str) -> float:
    if isinstance(value, Quantity):
        return float(value.to_value(target))
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"{field}: expected a Quantity or a number, got {type(value).__name__}")


def _miepython(
    m: complex, diameter: float, wavelength: float, medium: float
) -> dict[str, float]:
    import miepython

    # miepython returns (qext, qsca, qback, g) - the third value is the
    # backscatter efficiency, not absorption.  Absorption has to be derived
    # from the extinction budget; PyMieScatt reports it directly.
    qext, qsca, _qback, g = miepython.efficiencies(
        complex(m), diameter, wavelength, n_env=medium
    )
    return {
        "Qext": float(qext),
        "Qsca": float(qsca),
        "Qabs": float(qext - qsca),
        "g": float(g),
    }


def _pymiescatt(
    m: complex, diameter: float, wavelength: float, medium: float
) -> dict[str, float]:
    import PyMieScatt

    # PyMieScatt source and wheel artifacts carrying version 1.8.1.1 differ:
    # one scales only m by nMedium, while the other scales m and wavelength.
    # Normalise both physical inputs here and pass nMedium=1 so either artifact
    # evaluates the same size parameter and relative refractive index.
    result = PyMieScatt.MieQ(
        complex(m) / medium,
        wavelength / medium,
        diameter,
        nMedium=1.0,
        asDict=True,
    )
    return {key: float(result[key]) for key in _KEYS}


def _reference(
    m: complex, diameter: float, wavelength: float, medium: float
) -> dict[str, float]:
    from .reference_mie import mie_efficiencies_reference

    return mie_efficiencies_reference(m, diameter, wavelength, n_env=medium)


_IMPLEMENTATIONS = {
    "miepython": _miepython,
    "PyMieScatt": _pymiescatt,
    "optcon_reference": _reference,
}


def mie_efficiencies(
    *,
    m: complex,
    diameter: Any,
    wavelength: Any,
    medium_index: float = 1.0,
    engine: str = "miepython",
) -> dict[str, Quantity]:
    """Mie efficiencies for one homogeneous sphere.

    Returns ``Qext``, ``Qsca``, ``Qabs`` and the asymmetry parameter ``g`` as
    dimensionless quantities.  Both geometric inputs are keyword-only, so the
    ``(diameter, wavelength)`` versus ``(wavelength, diameter)`` split between
    engines cannot turn into a silent swap.

    ``wavelength`` is the **vacuum** wavelength and ``m`` is the index
    relative to vacuum, whatever the surrounding medium is.  ``medium_index``
    is always part of the adapter query; no external-library default is used.
    """
    spec = require_available(engine, "mie")
    implementation = _IMPLEMENTATIONS.get(engine)
    if implementation is None:
        raise ValueError(f"no Mie adapter implemented for {engine!r}")
    if not medium_index > 0.0:
        raise ValueError(f"medium_index must be positive, got {medium_index!r}")

    native = spec.length_unit
    d_native = _length_in(diameter, native, "diameter")
    l_native = _length_in(wavelength, native, "wavelength")
    values = implementation(m, d_native, l_native, float(medium_index))
    return {key: Quantity(values[key], _unit("1"), RATIO) for key in _KEYS}


__all__ = ["mie_efficiencies"]
