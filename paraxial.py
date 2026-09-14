"""Paraxial focal length of a singlet, computed by several engines.

This is the ray-tracing member of the adapter family: the same physical lens,
described once in units, evaluated either by optcon's own ABCD chain or by an
external lens-design package.  It exists so that the differential harness can
be pointed at ray tracers interactively, not only inside the test suite.
"""

from __future__ import annotations

import math
from typing import Any

from .elements import compose, effective_focal_length, free_space, refracting_surface
from .engines.registry import require_available
from .errors import DimensionError
from .quantity import Quantity
from .units import Unit
from .units import unit as _unit

_LENGTH = _unit("m").dimension


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(100, 'mm'), "
        f"got a bare {type(value).__name__}"
    )


def _optcon_paraxial(radius: float, thickness: float, index: float) -> float:
    """The ABCD chain: two curved interfaces separated by the glass path."""
    lens = compose(
        refracting_surface(1.0, index, Quantity(radius, _unit("mm"))),
        free_space(Quantity(thickness, _unit("mm"))),
        refracting_surface(index, 1.0, Quantity(-radius, _unit("mm"))),
    )
    return effective_focal_length(lens).to_value("mm")


def _optiland(radius: float, thickness: float, index: float) -> float:
    """The same lens through optiland's own paraxial solver."""
    import numpy as np
    from optiland.materials import IdealMaterial
    from optiland.optic import Optic

    glass = IdealMaterial(n=float(index), k=0.0)
    lens = Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=100.0)
    lens.surfaces.add(index=1, radius=radius, thickness=thickness, material=glass, is_stop=True)
    lens.surfaces.add(index=2, radius=-radius, thickness=50.0, material="air")
    lens.surfaces.add(index=3)
    lens.wavelengths.add(0.55, is_primary=True)
    return float(np.asarray(lens.paraxial.f2()).ravel()[0])


_IMPLEMENTATIONS = {
    "optcon_paraxial": _optcon_paraxial,
    "optiland": _optiland,
}


def thick_lens_focal_length(
    *,
    radius: Any,
    thickness: Any,
    refractive_index: float,
    engine: str = "optcon_paraxial",
) -> Quantity:
    """Effective focal length of a symmetric biconvex singlet.

    ``radius`` and ``thickness`` cross the boundary as lengths; only the
    millimetre magnitude matters to the engines, so the result is returned in
    millimetres.
    """
    spec = require_available(engine, "ray_tracing")
    implementation = _IMPLEMENTATIONS.get(engine)
    if implementation is None:
        raise ValueError(f"no paraxial adapter implemented for {engine!r}")
    del spec

    radius_mm, _ = _length_of(radius, "thick_lens_focal_length")
    thickness_mm, _ = _length_of(thickness, "thick_lens_focal_length")
    index = float(refractive_index)
    if index <= 1.0:
        raise ValueError(
            f"a lens must be optically denser than its surroundings, got n = {index}"
        )
    if thickness_mm <= 0.0:
        raise ValueError("a thick lens needs a positive centre thickness")
    if radius_mm == 0.0 or not math.isfinite(radius_mm):
        raise ValueError("the radius of curvature must be finite and non-zero")

    return Quantity(implementation(radius_mm, thickness_mm, index), _unit("mm"))


__all__ = ["thick_lens_focal_length"]
