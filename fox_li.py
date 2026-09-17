"""Fox-Li cavity diffraction integral operator for open resonators.

This module solves a Fredholm integral-operator eigenvalue problem for optical
resonators with finite mirror apertures:

    gamma * u(x_2) = \\int_{-a_1}^{a_1} K(x_1, x_2) u(x_1) dx_1

following Fox & Li, Bell Syst. Tech. J. 40, 453-488 (1961) and Siegman,
*Lasers* (1986), ch. 14 and 20.

Unlike paraxial ABCD ray matrices and Hermite-Gauss expansions that assume
infinite apertures (predicting zero diffraction loss), the Fox-Li operator
captures finite-aperture diffraction clipping, edge diffraction phase shifts,
transverse mode discrimination, and alignment sensitivity.

Physical invariants checked:
* Passivity: diffraction is strictly lossy; the operator satisfies
  sigma_max(K) <= 1, and every mode eigenvalue satisfies |gamma| <= 1.
* Reciprocity: for symmetric resonators (g1 = g2) without mirror tilt,
  the Fredholm kernel satisfies K(x1, x2) = K(x2, x1).
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as _la
import scipy.special as _sp

from .checks import assert_passive, assert_reciprocal
from .errors import DimensionError
from .quantity import Quantity
from .units import Unit
from .units import unit as _unit

_LENGTH = _unit("m").dimension
_ANGLE = _unit("rad").dimension


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        scalar = float(value.value)
        if not math.isfinite(scalar):
            raise ValueError(f"{context}: length must be finite, got {value}")
        return scalar, value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(100, 'mm'), "
        f"got a bare {type(value).__name__}"
    )


def _angle_of(value: Any, context: str) -> float:
    if value is None:
        return 0.0
    if isinstance(value, Quantity):
        if value.unit.dimension != _ANGLE:
            raise DimensionError(f"{context}: expected an angle, got {value.unit}")
        scalar = float(value.to_value("rad"))
        if not math.isfinite(scalar):
            raise ValueError(f"{context}: angle must be finite, got {value}")
        return scalar
    raise TypeError(
        f"{context}: expected an angle with units, e.g. q(100, 'urad'), "
        f"got a bare {type(value).__name__}"
    )


def _integer_at_least(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    integer = int(value)
    if integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return integer


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return integer


@dataclass(frozen=True)
class FoxLiResonator:
    """An open optical resonator with finite aperture mirrors.

    Parameters
    ----------
    wavelength : Quantity
        Optical wavelength lambda.
    length : Quantity
        Resonator optical cavity length L.
    aperture : Quantity
        Mirror half-width (for strip geometry) or mirror radius (for circular).
    g1 : float, default 1.0
        Cavity g-parameter for mirror 1: g1 = 1 - L / R1 (1.0 = flat mirror).
    g2 : float, default 1.0
        Cavity g-parameter for mirror 2: g2 = 1 - L / R2 (1.0 = flat mirror).
    tilt : Quantity | None, default None
        Angular misalignment tilt of mirror 1 (e.g. q(50, "urad")).
    geometry : str, default "strip"
        Aperture geometry: "strip" (1D Cartesian slit) or "circular" (2D axisymmetric).
    """

    wavelength: Quantity
    length: Quantity
    aperture: Quantity
    g1: float = 1.0
    g2: float = 1.0
    tilt: Quantity | None = None
    geometry: str = "strip"

    def __post_init__(self) -> None:
        lam_val, _ = _length_of(self.wavelength, "FoxLiResonator.wavelength")
        l_val, _ = _length_of(self.length, "FoxLiResonator.length")
        a_val, _ = _length_of(self.aperture, "FoxLiResonator.aperture")
        if lam_val <= 0.0 or l_val <= 0.0 or a_val <= 0.0:
            raise ValueError("wavelength, length, and aperture must be positive")
        for name, value in (("g1", self.g1), ("g2", self.g2)):
            try:
                scalar = float(value)
            except (TypeError, ValueError) as error:
                raise TypeError(f"{name} must be a finite real number") from error
            if not math.isfinite(scalar):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.geometry not in {"strip", "circular"}:
            raise ValueError(
                f"geometry must be 'strip' or 'circular', got {self.geometry!r}"
            )
        _angle_of(self.tilt, "FoxLiResonator.tilt")


def fresnel_number(resonator: FoxLiResonator) -> float:
    """Fresnel number N_F = a^2 / (lambda * L) of the open resonator."""
    lam_m = float(resonator.wavelength.to_value("m"))
    l_m = float(resonator.length.to_value("m"))
    a_m = float(resonator.aperture.to_value("m"))
    return (a_m * a_m) / (lam_m * l_m)


@dataclass(frozen=True)
class FoxLiMode:
    """A converged transverse eigenmode of the Fox-Li integral operator."""

    order: int
    azimuthal_order: int
    eigenvalue: complex
    loss: float
    round_trip_loss: float
    phase_shift: float
    coordinates: np.ndarray
    amplitude: np.ndarray
    fresnel_number: float


def fox_li_operator(
    resonator: FoxLiResonator,
    num_points: int = 128,
    azimuthal_order: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the discretized Nystrom Fredholm integral operator matrix.

    Returns
    -------
    K_sym : np.ndarray
        The symmetric quadrature-weighted kernel matrix (N x N), suitable for
        direct eigenvalue decomposition and invariant contract assertions.
    coords : np.ndarray
        Physical coordinate points along the mirror aperture (in SI metres).
    weights : np.ndarray
        Physical quadrature integration weights.
    """
    num_points = _integer_at_least(num_points, "num_points", 16)
    azimuthal_order = _nonnegative_integer(azimuthal_order, "azimuthal_order")

    n_f = fresnel_number(resonator)
    a_m = float(resonator.aperture.to_value("m"))
    lam_m = float(resonator.wavelength.to_value("m"))
    tilt_rad = _angle_of(resonator.tilt, "fox_li_operator")
    k = 2.0 * math.pi / lam_m

    if resonator.geometry == "strip":
        # Gauss-Legendre quadrature on [-1, 1]
        xi, w_xi = _sp.roots_legendre(num_points)
        coords = xi * a_m
        weights = w_xi * a_m

        # Dimensionless kernel in normalized coordinates xi in [-1, 1]
        xi_col = xi[:, None]
        xi_row = xi[None, :]

        # Phase factor from curvature g1, g2
        phase_curv = -1.0j * math.pi * n_f * (
            resonator.g1 * xi_col**2 - 2.0 * xi_col * xi_row + resonator.g2 * xi_row**2
        )
        kernel = np.sqrt(1.0j * n_f) * np.exp(phase_curv)

        if tilt_rad != 0.0:
            # Linear phase shift on mirror 1: exp(-i k theta x1)
            tilt_phase = np.exp(-1.0j * k * tilt_rad * coords)[:, None]
            kernel = kernel * tilt_phase

        # Symmetric quadrature weighting: K_sym_ij = sqrt(w_xi_i) * kernel_ij * sqrt(w_xi_j)
        w_sqrt = np.sqrt(w_xi)
        k_sym = w_sqrt[:, None] * kernel * w_sqrt[None, :]
        return k_sym, coords, weights

    else:
        # Circular geometry: Gauss-Legendre on rho in [0, 1]
        m = azimuthal_order
        xi_std, w_std = _sp.roots_legendre(num_points)
        rho = 0.5 * (xi_std + 1.0)
        w_rho = 0.5 * w_std
        coords = rho * a_m
        # Integration measure is 2 pi r dr = 2 pi a^2 rho drho
        weights = 2.0 * math.pi * (a_m**2) * rho * w_rho

        rho_col = rho[:, None]
        rho_row = rho[None, :]

        arg_bessel = 2.0 * math.pi * n_f * rho_col * rho_row
        j_m = _sp.jv(m, arg_bessel)

        phase_curv = -1.0j * math.pi * n_f * (
            resonator.g1 * rho_col**2 + resonator.g2 * rho_row**2
        )
        # Prefactor: 2 pi * i^(m+1) * N_F * sqrt(rho1 * rho2)
        prefactor = 2.0 * math.pi * (1.0j ** (m + 1)) * n_f * np.sqrt(rho_col * rho_row)
        kernel = prefactor * j_m * np.exp(phase_curv)

        w_sqrt = np.sqrt(w_rho)
        k_sym = w_sqrt[:, None] * kernel * w_sqrt[None, :]
        return k_sym, coords, weights


def solve_fox_li_modes(
    resonator: FoxLiResonator,
    num_modes: int = 4,
    num_points: int = 128,
    azimuthal_order: int = 0,
    check_contracts: bool = True,
) -> list[FoxLiMode]:
    """Solve for the lowest-loss transverse eigenmodes of the open resonator.

    Parameters
    ----------
    resonator : FoxLiResonator
        The resonator configuration.
    num_modes : int, default 4
        Number of lowest-loss transverse modes to return.
    num_points : int, default 128
        Number of quadrature discretization points.
    azimuthal_order : int, default 0
        Azimuthal mode index m (for circular geometry).
    check_contracts : bool, default True
        If True, validates passivity and reciprocity contracts on the operator.

    Returns
    -------
    list[FoxLiMode]
        Modes ordered by diffraction loss ascending (fundamental mode first).
    """
    num_modes = _integer_at_least(num_modes, "num_modes", 1)
    k_sym, coords, weights = fox_li_operator(
        resonator, num_points=num_points, azimuthal_order=azimuthal_order
    )

    if check_contracts:
        # Passivity: diffraction cannot amplify energy
        assert_passive(k_sym, rtol=1e-5, atol=1e-5, name="fox_li_operator")
        # Reciprocity: symmetric cavity without tilt is reciprocal
        if resonator.g1 == resonator.g2 and (
            resonator.tilt is None or _angle_of(resonator.tilt, "") == 0.0
        ):
            assert_reciprocal(k_sym, rtol=1e-5, atol=1e-5, name="fox_li_operator")

    evals, evecs = _la.eig(k_sym)
    n_f = fresnel_number(resonator)

    # Sort by loss ascending (|gamma| descending)
    idx = np.argsort(-np.abs(evals))
    evals = evals[idx]
    evecs = evecs[:, idx]

    modes: list[FoxLiMode] = []
    limit = min(num_modes, len(evals))

    if resonator.geometry == "strip":
        _, w_xi = _sp.roots_legendre(num_points)
        w_sqrt = np.sqrt(w_xi)
    else:
        _, w_std = _sp.roots_legendre(num_points)
        rho = 0.5 * (_sp.roots_legendre(num_points)[0] + 1.0)
        w_rho = 0.5 * w_std
        w_sqrt = np.sqrt(w_rho)

    for i in range(limit):
        gamma = evals[i]
        mag = abs(gamma)
        loss = max(0.0, 1.0 - mag * mag)
        rt_loss = max(0.0, 1.0 - mag**4)
        phase = cmath.phase(gamma)

        # Recover physical eigenfunction from symmetric quadrature vector
        v = evecs[:, i]
        if resonator.geometry == "strip":
            u = v / w_sqrt
            norm = math.sqrt(float(np.sum(weights * np.abs(u) ** 2)))
            if norm > 0.0:
                u = u / norm
        else:
            # For circular geometry, v corresponds to sqrt(rho) * u(rho)
            u = (v / w_sqrt) / np.sqrt(np.maximum(rho, 1e-12))
            norm = math.sqrt(float(np.sum(weights * np.abs(u) ** 2)))
            if norm > 0.0:
                u = u / norm

        # Fix arbitrary global phase so central amplitude has non-negative real part
        mid = len(u) // 2
        ref_phase = cmath.phase(u[mid]) if abs(u[mid]) > 1e-6 else 0.0
        u = u * np.exp(-1.0j * ref_phase)

        modes.append(
            FoxLiMode(
                order=i,
                azimuthal_order=int(azimuthal_order),
                eigenvalue=complex(gamma),
                loss=float(loss),
                round_trip_loss=float(rt_loss),
                phase_shift=float(phase),
                coordinates=coords,
                amplitude=u,
                fresnel_number=float(n_f),
            )
        )

    return modes


def fox_li_power_iteration(
    resonator: FoxLiResonator,
    initial_field: np.ndarray | None = None,
    max_iter: int = 300,
    tol: float = 1e-7,
    num_points: int = 128,
    azimuthal_order: int = 0,
) -> tuple[FoxLiMode, dict[str, list[float]]]:
    """Simulate physical wave relaxation inside the open resonator via power iteration.

    Parameters
    ----------
    resonator : FoxLiResonator
        The resonator configuration.
    initial_field : np.ndarray | None, default None
        Initial transverse field amplitude on the mirror aperture.
        If None, initializes with a uniform field.
    max_iter : int, default 300
        Maximum round-trip transits.
    tol : float, default 1e-7
        Convergence tolerance on eigenvalue difference |gamma_{k} - gamma_{k-1}|.
    num_points : int, default 128
        Number of quadrature discretization points.
    azimuthal_order : int, default 0
        Azimuthal mode index m (for circular geometry).

    Returns
    -------
    mode : FoxLiMode
        The converged fundamental transverse eigenmode.
    history : dict[str, list[float]]
        Iteration history containing 'loss', 'phase_shift', and 'error'.
    """
    max_iter = _integer_at_least(max_iter, "max_iter", 1)
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(
            f"tol must be finite and strictly positive, got {tol!r}"
        )

    k_sym, coords, weights = fox_li_operator(
        resonator, num_points=num_points, azimuthal_order=azimuthal_order
    )
    n_f = fresnel_number(resonator)

    if resonator.geometry == "strip":
        _, w_xi = _sp.roots_legendre(num_points)
        w_sqrt = np.sqrt(w_xi)
    else:
        _, w_std = _sp.roots_legendre(num_points)
        rho = 0.5 * (_sp.roots_legendre(num_points)[0] + 1.0)
        w_sqrt = np.sqrt(0.5 * w_std)

    if initial_field is None:
        u_init = np.ones(num_points, dtype=complex)
    else:
        u_init = np.asarray(initial_field, dtype=complex)
        if u_init.shape != (num_points,):
            raise ValueError(
                f"initial_field must have shape ({num_points},), got {u_init.shape}"
            )
        if not np.isfinite(u_init).all():
            raise ValueError("initial_field must contain only finite values")

    if resonator.geometry == "strip":
        v = w_sqrt * u_init
    else:
        v = w_sqrt * np.sqrt(np.maximum(rho, 1e-12)) * u_init

    v_norm = np.linalg.norm(v)
    if not math.isfinite(float(v_norm)) or v_norm == 0.0:
        raise ValueError("initial field cannot be identically zero")
    v = v / v_norm

    history: dict[str, list[float]] = {
        "loss": [],
        "phase_shift": [],
        "error": [],
    }

    gamma_prev = 0.0 + 0.0j

    for _ in range(max_iter):
        v_next = k_sym @ v
        gamma = complex(np.vdot(v, v_next) / np.vdot(v, v))
        v_norm = float(np.linalg.norm(v_next))
        if v_norm == 0.0:
            raise RuntimeError("power iteration collapsed to zero amplitude")
        v_next = v_next / v_norm

        err = abs(gamma - gamma_prev)
        history["loss"].append(float(max(0.0, 1.0 - abs(gamma) ** 2)))
        history["phase_shift"].append(float(cmath.phase(gamma)))
        history["error"].append(float(err))

        gamma_prev = gamma
        v = v_next
        if err < tol:
            break

    mag = abs(gamma)
    loss = max(0.0, 1.0 - mag * mag)
    rt_loss = max(0.0, 1.0 - mag**4)
    phase = cmath.phase(gamma)

    if resonator.geometry == "strip":
        u = v / w_sqrt
    else:
        u = (v / w_sqrt) / np.sqrt(np.maximum(rho, 1e-12))

    norm = math.sqrt(float(np.sum(weights * np.abs(u) ** 2)))
    if norm > 0.0:
        u = u / norm

    mid = len(u) // 2
    ref_phase = cmath.phase(u[mid]) if abs(u[mid]) > 1e-6 else 0.0
    u = u * np.exp(-1.0j * ref_phase)

    mode = FoxLiMode(
        order=0,
        azimuthal_order=int(azimuthal_order),
        eigenvalue=gamma,
        loss=float(loss),
        round_trip_loss=float(rt_loss),
        phase_shift=float(phase),
        coordinates=coords,
        amplitude=u,
        fresnel_number=float(n_f),
    )

    return mode, history


__all__ = [
    "FoxLiMode",
    "FoxLiResonator",
    "fox_li_operator",
    "fox_li_power_iteration",
    "fresnel_number",
    "solve_fox_li_modes",
]
