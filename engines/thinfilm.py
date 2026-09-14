"""Typed thin-film response on top of the surveyed TMM engines."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ..errors import DimensionError
from ..quantity import POWER, Quantity, q
from ..units import unit as _unit
from .registry import engine_spec, import_engine, require_available


def _length_in(value: Any, target: str, field: str) -> float:
    """Interpret a bare number in the engine's own unit, or convert a Quantity."""
    if isinstance(value, Quantity):
        return float(value.to_value(target))
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"{field}: expected a Quantity or a number, got {type(value).__name__}")


def _angle_in_radians(value: Any) -> float:
    if isinstance(value, Quantity):
        if value.unit.dimension != _unit("rad").dimension:
            raise DimensionError(f"angle must have angle dimension, got {value.unit}")
        return float(value.to_value("rad"))
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"angle: expected a Quantity or a number, got {type(value).__name__}")


def _tmm_core(n_list, d_list, theta_rad, wavelength, polarization):
    import tmm_core

    result = tmm_core.coh_tmm(polarization, n_list, d_list, theta_rad, wavelength)
    return float(result["R"]), float(result["T"])


def _tmm_fast(n_list, d_list, theta_rad, wavelength, polarization):
    import tmm_fast

    layers = len(n_list)
    indices = np.asarray(n_list, dtype=complex).reshape(1, layers, 1)
    thicknesses = np.asarray(d_list, dtype=float).reshape(1, layers)
    angles = np.asarray([theta_rad], dtype=float)
    wavelengths = np.asarray([wavelength], dtype=float)
    result = tmm_fast.coh_tmm(polarization, indices, thicknesses, angles, wavelengths)
    reflectance = float(np.asarray(result["R"]).reshape(-1)[0])
    transmittance = float(np.asarray(result["T"]).reshape(-1)[0])
    return reflectance, transmittance


_IMPLEMENTATIONS = {"tmm_core": _tmm_core, "tmm_fast": _tmm_fast}


def stack_response(
    *,
    n_list: Sequence[complex],
    thicknesses: Sequence[Any],
    wavelength: Any,
    angle: Any,
    polarization: str = "s",
    engine: str = "tmm_core",
) -> dict[str, Quantity]:
    """Reflectance and transmittance of a layer stack, as typed power ratios.

    ``thicknesses`` may mix numbers (taken in the engine's native length unit)
    and Quantities such as ``q(100, "nm")``, which are converted for you.
    """
    spec = require_available(engine, "thin_film")
    implementation = _IMPLEMENTATIONS.get(engine)
    if implementation is None:
        raise ValueError(f"no thin-film adapter implemented for {engine!r}")

    if polarization not in {"s", "p"}:
        raise ValueError(f"polarization must be 's' or 'p', got {polarization!r}")
    if len(n_list) != len(thicknesses):
        raise ValueError(
            f"n_list has {len(n_list)} layers but thicknesses has {len(thicknesses)}"
        )

    native_unit = spec.length_unit
    converted = []
    for index, value in enumerate(thicknesses):
        if isinstance(value, float) and math.isinf(value):
            converted.append(math.inf)
        else:
            converted.append(_length_in(value, native_unit, f"thicknesses[{index}]"))
    native_wavelength = _length_in(wavelength, native_unit, "wavelength")
    theta_radians = _angle_in_radians(angle)

    reflectance, transmittance = implementation(
        list(n_list), converted, theta_radians, native_wavelength, polarization
    )
    return {
        "R": Quantity(reflectance, _unit("1"), POWER),
        "T": Quantity(transmittance, _unit("1"), POWER),
    }


def engine_units(engine: str) -> dict[str, str]:
    """Conventions of one thin-film engine, for reporting."""
    spec = engine_spec(engine)
    return {"length": spec.length_unit, "angle": spec.angle_unit}


__all__ = ["stack_response", "engine_units", "import_engine", "q"]
