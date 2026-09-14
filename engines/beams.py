"""Beam-propagation engines, including a closed-form reference.

The reference here is not another library - it is the textbook Gaussian beam
relation ``w(z) = w0 sqrt(1 + (z/zR)^2)``.  An engine disagreeing with it is
not a matter of opinion, which makes this the sharpest differential test in
the package.

The two LightPipes commands are registered as separate engines because they
are separate propagators with different accuracy, and the whole point is to
be able to tell them apart.
"""

from __future__ import annotations

import math
from typing import Any

from ..quantity import Quantity
from ..units import unit as _unit
from .registry import require_available

DEFAULT_SAMPLES = 512


def _length_in(value: Any, target: str, field: str) -> float:
    if isinstance(value, Quantity):
        return float(value.to_value(target))
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"{field}: expected a Quantity or a number, got {type(value).__name__}")


def _analytic(waist: float, wavelength: float, distance: float, **_kwargs) -> float:
    rayleigh = math.pi * waist**2 / wavelength
    return waist * math.sqrt(1.0 + (distance / rayleigh) ** 2)


def _lightpipes(
    waist: float,
    wavelength: float,
    distance: float,
    *,
    grid_size: float,
    samples: int,
    command: str,
) -> float:
    import LightPipes

    field = LightPipes.Begin(grid_size, wavelength, samples)
    field = LightPipes.GaussBeam(field, waist)
    if distance:
        field = getattr(LightPipes, command)(field, distance)
    # LightPipes' D4sigma returns the 4-sigma width, i.e. twice the 1/e^2 radius
    return float(LightPipes.D4sigma(field)[0]) / 2.0


def _forvard(waist, wavelength, distance, *, grid_size, samples):
    return _lightpipes(
        waist, wavelength, distance, grid_size=grid_size, samples=samples, command="Forvard"
    )


def _fresnel(waist, wavelength, distance, *, grid_size, samples):
    return _lightpipes(
        waist, wavelength, distance, grid_size=grid_size, samples=samples, command="Fresnel"
    )


def _diffractio(waist, wavelength, distance, *, grid_size, samples):
    """Chirp-z propagation through diffractio, which works in microns."""
    import numpy as np
    from diffractio import um
    from diffractio.scalar_fields_XY import Scalar_field_XY

    del um  # the module's constants are 1.0 by definition; the factor is below
    to_microns = 1000.0  # our values are millimetres, diffractio's base is um
    waist_um = waist * to_microns
    wavelength_um = wavelength * to_microns
    extent_um = grid_size * to_microns
    spacing_um = extent_um / samples
    # CZT/RS integrate a Rayleigh-Sommerfeld kernel, which needs the field
    # sampled at roughly one point per wavelength.  Measured here: workable up
    # to about 1.5 wavelengths per sample, unusable at 37 (where the answer
    # came out three orders of magnitude wrong rather than merely imprecise).
    if spacing_um > 1.5 * wavelength_um:
        raise ValueError(
            "diffractio's CZT needs near-wavelength sampling: this grid has "
            f"{spacing_um / wavelength_um:.1f} wavelengths per sample, and the "
            "method is only valid to about 1.5. Pass a finer grid_size/samples."
        )
    axis = np.linspace(-extent_um / 2.0, extent_um / 2.0, samples)
    field = Scalar_field_XY(x=axis, y=axis, wavelength=wavelength_um)
    grid_x, grid_y = np.meshgrid(axis, axis)
    field.u = np.exp(-(grid_x**2 + grid_y**2) / waist_um**2).astype(complex)
    # CZT returns the propagated field; ignoring the return value is the
    # documented way to get a silent no-op here
    moved = field.CZT(z=distance * to_microns) if distance else field
    values = np.abs(np.asarray(moved.u if hasattr(moved, "u") else moved)) ** 2
    # CZT re-derives the output plane's sampling, so the radius has to be
    # measured on the grid that came back rather than on the one that went in
    output_axis = np.asarray(getattr(moved, "x", axis), dtype=float)
    marginal = values.sum(axis=1)
    centroid = float((marginal * output_axis).sum() / marginal.sum())
    variance = float(
        (marginal * (output_axis - centroid) ** 2).sum() / marginal.sum()
    )
    return 2.0 * math.sqrt(variance) / to_microns


_IMPLEMENTATIONS = {
    "optcon_beam_reference": _analytic,
    "lightpipes_forvard": _forvard,
    "lightpipes_fresnel": _fresnel,
    "diffractio": _diffractio,
}


def gaussian_beam_radius(
    *,
    waist: Any,
    wavelength: Any,
    distance: Any,
    grid_size: Any = None,
    samples: int = DEFAULT_SAMPLES,
    engine: str = "lightpipes_forvard",
) -> dict[str, Quantity]:
    """1/e^2 radius after propagating ``distance`` from the waist.

    All three inputs cross the boundary as typed quantities; ``grid_size``
    defaults to twenty waists, which is ample for a beam that starts at a
    waist.  The result carries the engine's native length unit.
    """
    spec = require_available(engine, "beam")
    implementation = _IMPLEMENTATIONS.get(engine)
    if implementation is None:
        raise ValueError(f"no beam adapter implemented for {engine!r}")

    native = spec.length_unit
    waist_native = _length_in(waist, native, "waist")
    wavelength_native = _length_in(wavelength, native, "wavelength")
    distance_native = _length_in(distance, native, "distance")
    if waist_native <= 0.0 or wavelength_native <= 0.0:
        raise ValueError("waist and wavelength must be positive")
    grid_native = (
        20.0 * waist_native
        if grid_size is None
        else _length_in(grid_size, native, "grid_size")
    )

    width = implementation(
        waist_native,
        wavelength_native,
        distance_native,
        grid_size=grid_native,
        samples=int(samples),
    )
    return {"w": Quantity(float(width), _unit(native))}


__all__ = ["gaussian_beam_radius"]
