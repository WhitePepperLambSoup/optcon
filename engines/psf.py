"""Airy radius of a circular pupil, computed by several engines.

The diffraction limit is one of the few quantities in optics with an exact
answer, which makes it the ideal thing to point a wavefront-propagation
package at: ``theta = 1.22 lambda / D`` needs no adjudication.

``optcon_diffraction`` evaluates that closed form; ``poppy`` builds an optical
system, propagates the wavefront and measures the first dark ring of the
resulting point spread function.
"""

from __future__ import annotations

import math
from typing import Any

from ..diffraction import angular_resolution
from ..errors import DimensionError
from ..quantity import Quantity
from ..units import Unit
from ..units import unit as _unit
from .registry import require_available

_LENGTH = _unit("m").dimension
_ARCSEC = math.pi / 180.0 / 3600.0  # one arcsecond in radians

#: grid used for the poppy wavefront; fine enough to place ~20 samples
#: inside the first Airy lobe
_POPPY_SAMPLES = 256
_POPPY_OVERSAMPLE = 4


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(1, 'm'), "
        f"got a bare {type(value).__name__}"
    )


def _optcon_diffraction(aperture_m: float, wavelength_m: float) -> float:
    angle = angular_resolution(
        Quantity(wavelength_m, _unit("m")), Quantity(aperture_m, _unit("m"))
    )
    return angle.to_value("rad")


def _poppy(aperture_m: float, wavelength_m: float) -> float:
    import astropy.units as u
    import numpy as np
    import poppy

    expected_arcsec = 1.22 * wavelength_m / aperture_m / _ARCSEC
    pixel_scale = expected_arcsec / 20.0
    system = poppy.OpticalSystem(
        npix=_POPPY_SAMPLES, oversample=_POPPY_OVERSAMPLE
    )
    system.add_pupil(poppy.CircularAperture(radius=(aperture_m / 2.0) * u.m))
    system.add_detector(pixelscale=pixel_scale, fov_arcsec=4.0 * expected_arcsec)
    psf = system.calc_psf(wavelength=wavelength_m)
    image = np.asarray(psf[0].data, dtype=float)
    # poppy re-derives the detector sampling from the requested field of view,
    # so the pixel scale that actually applies is the one in the header - the
    # value passed in above is a request, not a fact
    actual_scale = float(psf[0].header["PIXELSCL"])
    centre = image.shape[0] // 2
    profile = image[centre, centre:]
    # walk out from the peak to the first local minimum
    for index in range(1, profile.size - 1):
        if profile[index] < profile[index - 1] and profile[index] <= profile[index + 1]:
            return float(index * actual_scale * _ARCSEC)
    raise RuntimeError("no first dark ring was found in the point spread function")


_IMPLEMENTATIONS = {
    "optcon_diffraction": _optcon_diffraction,
    "poppy": _poppy,
}


def airy_psf_radius(
    *, aperture_diameter: Any, wavelength: Any, engine: str = "optcon_diffraction"
) -> Quantity:
    """Angular radius of the first dark ring of the Airy pattern, in radians."""
    require_available(engine, "diffraction")
    implementation = _IMPLEMENTATIONS.get(engine)
    if implementation is None:
        raise ValueError(f"no Airy adapter implemented for {engine!r}")
    diameter_magnitude, diameter_unit = _length_of(
        aperture_diameter, "airy_psf_radius"
    )
    wavelength_magnitude, wavelength_unit = _length_of(wavelength, "airy_psf_radius")
    diameter_m = diameter_unit.to_value(diameter_magnitude, _unit("m"))
    wavelength_m = wavelength_unit.to_value(wavelength_magnitude, _unit("m"))
    if diameter_m <= 0.0 or wavelength_m <= 0.0:
        raise ValueError("the aperture diameter and wavelength must be positive")
    return Quantity(implementation(diameter_m, wavelength_m), _unit("rad"))


__all__ = ["airy_psf_radius"]
