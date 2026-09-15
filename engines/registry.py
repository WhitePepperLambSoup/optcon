"""Where the optical engines live and what units they speak.

Every engine in this workspace was written with its own convention - one
works in nanometres, another in SI metres, one takes ``(diameter,
wavelength)`` and another ``(wavelength, diameter)``.  None of that is
wrong; it is simply invisible.  This registry makes the convention explicit
data, so the adapters can convert at the boundary instead of relying on the
caller remembering.

The engine *sources* are not vendored into optcon: they stay in the corpus
directories already present in the workspace, resolved relative to the
project root (override with ``OPTCON_CORPUS_ROOT``).
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def corpus_root() -> Path:
    """Directory holding ``optical_simulation_projects/`` and friends."""
    override = os.environ.get("OPTCON_CORPUS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EngineSpec:
    """One external optical engine and the conventions it insists on."""

    name: str
    domain: str
    module: str
    project_path: str
    length_unit: str
    angle_unit: str
    takes_polarization: bool
    summary: str


ENGINE_SPECS: dict[str, EngineSpec] = {
    "tmm_core": EngineSpec(
        name="tmm_core",
        domain="thin_film",
        module="tmm_core",
        project_path="optical_simulation_projects/tmm",
        length_unit="nm",
        angle_unit="rad",
        takes_polarization=True,
        summary=(
            "Steven Byrnes reference TMM (numpy, scalar); nm + radians; "
            "returns both r,t amplitudes and R,T powers"
        ),
    ),
    "tmm_fast": EngineSpec(
        name="tmm_fast",
        domain="thin_film",
        module="tmm_fast",
        project_path="ml_optical_projects/tmm_fast",
        length_unit="m",
        angle_unit="rad",
        takes_polarization=True,
        summary=(
            "vectorised differentiable TMM (numpy or torch); SI metres + radians; "
            "batched arrays only, one call handles many stacks"
        ),
    ),
    "miepython": EngineSpec(
        name="miepython",
        domain="mie",
        module="miepython",
        project_path="optical_simulation_projects/miepython",
        length_unit="nm",
        angle_unit="rad",
        takes_polarization=False,
        summary="Mie efficiencies with signature (m, diameter, wavelength)",
    ),
    "PyMieScatt": EngineSpec(
        name="PyMieScatt",
        domain="mie",
        module="PyMieScatt",
        project_path="optical_simulation_projects/PyMieScatt",
        length_unit="nm",
        angle_unit="rad",
        takes_polarization=False,
        summary="Mie efficiencies with signature (m, wavelength, diameter)",
    ),
    "optcon_reference": EngineSpec(
        name="optcon_reference",
        domain="mie",
        module="optcon.engines.reference_mie",
        project_path="",
        length_unit="nm",
        angle_unit="rad",
        takes_polarization=False,
        summary=(
            "optcon's own Mie series (SciPy Bessel/Hankel, no shared code with "
            "either library) used to adjudicate the other two"
        ),
    ),
    "optcon_beam_reference": EngineSpec(
        name="optcon_beam_reference",
        domain="beam",
        module="optcon.engines.beams",
        project_path="",
        length_unit="mm",
        angle_unit="rad",
        takes_polarization=False,
        summary="closed-form Gaussian beam w(z) = w0 sqrt(1 + (z/zR)^2)",
    ),
    "lightpipes_forvard": EngineSpec(
        name="lightpipes_forvard",
        domain="beam",
        module="LightPipes",
        project_path="optical_simulation_projects/lightpipes",
        length_unit="mm",
        angle_unit="rad",
        takes_polarization=False,
        summary="LightPipes spectral propagator (Forvard); examples work in mm",
    ),
    "lightpipes_fresnel": EngineSpec(
        name="lightpipes_fresnel",
        domain="beam",
        module="LightPipes",
        project_path="optical_simulation_projects/lightpipes",
        length_unit="mm",
        angle_unit="rad",
        takes_polarization=False,
        summary=(
            "LightPipes convolution propagator (Fresnel); same library, "
            "different propagator, several percent wide on a Gaussian"
        ),
    ),
    "diffractio": EngineSpec(
        name="diffractio",
        domain="beam",
        module="diffractio",
        project_path="optical_simulation_projects/diffractio",
        length_unit="um",
        angle_unit="rad",
        takes_polarization=False,
        summary=(
            "scalar/vector diffraction; ships its own unit constants "
            "(um=1.0, nm=0.001, mm=1000.0), so its internal base is the micron. "
            "Its propagators (CZT, RS, WPM) RETURN a new field rather than "
            "modifying in place, which silently looks like a no-op otherwise"
        ),
    ),
    "tracepy": EngineSpec(
        name="tracepy",
        domain="ray_tracing",
        module="tracepy",
        project_path="optical_simulation_projects/tracepy",
        length_unit="um",
        angle_unit="rad",
        takes_polarization=False,
        summary=(
            "Spencer-Murty sequential ray tracing; its RayGroup docstring "
            "states the wavelength is in microns"
        ),
    ),
    "pyoptools": EngineSpec(
        name="pyoptools",
        domain="ray_tracing",
        module="pyoptools",
        project_path="optical_simulation_projects/pyoptools",
        length_unit="mm",
        angle_unit="rad",
        takes_polarization=False,
        summary="Cython ray tracing and wavefronts; millimetre convention",
    ),
    "optiland": EngineSpec(
        name="optiland",
        domain="ray_tracing",
        module="optiland",
        project_path="ml_optical_projects/optiland",
        length_unit="mm",
        angle_unit="rad",
        takes_polarization=False,
        summary=(
            "differentiable lens design with a paraxial solver; millimetre "
            "lens units, wavelengths in micrometres"
        ),
    ),
    "rayoptics": EngineSpec(
        name="rayoptics",
        domain="ray_tracing",
        module="rayoptics",
        project_path="optical_simulation_projects/rayoptics/src",
        length_unit="mm",
        angle_unit="rad",
        takes_polarization=False,
        summary=(
            "imaging lens design in the Zemax tradition; src layout. The "
            "package imports, but the high-level OpticalSystem builder also "
            "pulls json_tricks, anytree, transforms3d, parsimonious, "
            "opticalglass and zmxtools, so it needs a full dependency install"
        ),
    ),
    "deeplens": EngineSpec(
        name="deeplens",
        domain="lens_design",
        module="deeplens",
        project_path="ml_optical_projects/deeplens",
        length_unit="mm",
        angle_unit="rad",
        takes_polarization=False,
        summary="end-to-end differentiable camera pipeline (PyTorch)",
    ),
    "poppy": EngineSpec(
        name="poppy",
        domain="diffraction",
        module="poppy",
        project_path="optical_simulation_projects/poppy",
        length_unit="m",
        angle_unit="rad",
        takes_polarization=False,
        summary=(
            "telescope PSF and physical-optics propagation; astropy units, "
            "so lengths cross the boundary as m and angles as rad or arcsec"
        ),
    ),
    "prysm": EngineSpec(
        name="prysm",
        domain="diffraction",
        module="prysm",
        project_path="optical_simulation_projects/prysm",
        length_unit="mm",
        angle_unit="rad",
        takes_polarization=False,
        summary=(
            "numerical optics with an astropy-based unit layer of its own. Its "
            "Wavefront.free_space could not be matched to a Fresnel "
            "propagation from the docstrings: at z = z_R it changes the field "
            "by ~1e-3 while a propagating beam needs ~1e10 times that, so no "
            "adapter ships until the convention is established"
        ),
    ),
    "neuroptica": EngineSpec(
        name="neuroptica",
        domain="photonic_circuit",
        module="neuroptica",
        project_path="ml_optical_projects/neuroptica",
        length_unit="um",
        angle_unit="rad",
        takes_polarization=False,
        summary=(
            "MZI mesh optical neural networks; ships an is_unitary helper "
            "that the checks module can be compared against"
        ),
    ),
    "ceviche": EngineSpec(
        name="ceviche",
        domain="em_solver",
        module="ceviche",
        project_path="ml_optical_projects/ceviche",
        length_unit="m",
        angle_unit="rad",
        takes_polarization=True,
        summary=(
            "FDFD/FDTD with an autograd adjoint; metre grid units, and its "
            "jacobian can be verified with checks.assert_gradient_matches"
        ),
    ),
    "femwell": EngineSpec(
        name="femwell",
        domain="em_solver",
        module="femwell",
        project_path="optical_simulation_projects/femwell",
        length_unit="um",
        angle_unit="rad",
        takes_polarization=True,
        summary="finite element mode solvers for photonic waveguides",
    ),
    "A_FMM": EngineSpec(
        name="A_FMM",
        domain="em_solver",
        module="A_FMM",
        project_path="optical_simulation_projects/A_FMM",
        length_unit="um",
        angle_unit="rad",
        takes_polarization=True,
        summary="aperiodic Fourier modal method (RCWA family) for gratings",
    ),
    "torchoptics": EngineSpec(
        name="torchoptics",
        domain="diffraction",
        module="torchoptics",
        project_path="ml_optical_projects/torchoptics/src",
        length_unit="m",
        angle_unit="rad",
        takes_polarization=True,
        summary=(
            "differentiable Fourier optics in PyTorch; the shallow clone is "
            "missing its generated _version module, so it is registered but "
            "not importable"
        ),
    ),
    "optcon_paraxial": EngineSpec(
        name="optcon_paraxial",
        domain="ray_tracing",
        module="optcon.paraxial",
        project_path="",
        length_unit="mm",
        angle_unit="rad",
        takes_polarization=False,
        summary="optcon's own ABCD chain, used as the reference for ray tracers",
    ),
    "optcon_diffraction": EngineSpec(
        name="optcon_diffraction",
        domain="diffraction",
        module="optcon.diffraction",
        project_path="",
        length_unit="m",
        angle_unit="rad",
        takes_polarization=False,
        summary="optcon's closed-form Airy radius, the reference for PSF engines",
    ),
}


def _prepare_import_compatibility(name: str) -> None:
    """Apply narrowly-scoped compatibility shims before importing an engine.

    PyMieScatt releases before 1.9 import ``scipy.integrate.trapz`` even though
    SciPy 1.14 removed that alias.  Restoring it as a process-local
    compatibility alias keeps the shim scoped to this optional import path
    while allowing the differential adapter to run on supported modern SciPy.
    """
    if name != "PyMieScatt":
        return
    try:
        import numpy as np
        import scipy.integrate as integrate
    except ImportError:
        return
    if not hasattr(integrate, "trapz"):
        trapezoid = getattr(np, "trapezoid", None)
        if trapezoid is None:
            trapezoid = getattr(np, "trapz", None)
        if trapezoid is not None:
            integrate.trapz = trapezoid


@lru_cache(maxsize=None)
def import_engine(name: str) -> tuple[bool, str]:
    """Return ``(importable, message)`` for one engine, adding its path once."""
    spec = engine_spec(name)
    if spec.project_path:
        path = corpus_root() / spec.project_path
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    _prepare_import_compatibility(name)
    try:
        importlib.import_module(spec.module)
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        return False, f"{type(error).__name__}: {error}"
    return True, ""


def _is_missing_engine(spec: EngineSpec, message: str) -> bool:
    """Return whether an import failure means the engine itself is absent."""
    return message.startswith("ModuleNotFoundError") and (
        f"No module named '{spec.module}'" in message
        or f'No module named "{spec.module}"' in message
    )


def engine_status(name: str) -> str:
    """Return ``ready``, ``missing`` or ``error`` for an engine import.

    ``missing`` means the optional package/source is not installed.  ``error``
    means an engine was found but failed during import, which must be visible in
    CI rather than being silently converted into a passing skip.
    """
    spec = engine_spec(name)
    ok, message = import_engine(name)
    if ok:
        return "ready"
    return "missing" if _is_missing_engine(spec, message) else "error"


def engine_spec(name: str) -> EngineSpec:
    try:
        return ENGINE_SPECS[name]
    except KeyError:
        known = ", ".join(sorted(ENGINE_SPECS))
        raise KeyError(f"unknown engine {name!r}; registered engines: {known}") from None


def is_available(name: str) -> bool:
    return import_engine(name)[0]


def available_engines(domain: str | None = None) -> list[str]:
    """Names of engines that actually import in this environment."""
    names = [
        name
        for name in ENGINE_SPECS
        if is_available(name)
        and (domain is None or ENGINE_SPECS[name].domain == domain)
    ]
    return sorted(names)


def describe_engines() -> str:
    """A human-readable table of engines, conventions and availability."""
    header = f"{'engine':12s} {'domain':10s} {'length':7s} {'angle':6s} {'status':9s} notes"
    lines = [header, "-" * len(header)]
    for name in sorted(ENGINE_SPECS):
        spec = ENGINE_SPECS[name]
        ok, message = import_engine(name)
        status = engine_status(name)
        note = spec.summary if ok else message
        lines.append(
            f"{spec.name:12s} {spec.domain:10s} {spec.length_unit:7s} "
            f"{spec.angle_unit:6s} {status:9s} {note}"
        )
    return "\n".join(lines)


def require_available(name: str, domain: str) -> EngineSpec:
    """Fetch a spec and fail loudly (with the import error) if unusable."""
    spec = engine_spec(name)
    if spec.domain != domain:
        raise ValueError(f"engine {name!r} handles {spec.domain}, not {domain}")
    ok, message = import_engine(name)
    if not ok:
        raise RuntimeError(f"engine {name!r} is not importable here: {message}")
    return spec
