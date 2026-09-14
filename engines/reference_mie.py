"""An independent Mie reference implementation.

Written straight from the standard Mie series (Bohren & Huffman, /Absorbing
Scattering of Light by Small Particles/, ch. 4) using SciPy's Bessel and
Hankel functions.  It shares no code path with the two libraries it is used
to adjudicate, so when three implementations disagree this one breaks the tie.

It is intentionally simple and slow: it exists to settle arguments, not to
run inside a training loop.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import hankel1, jv


def wiscombe_terms(x: float) -> int:
    """Wiscombe's truncation criterion for the Mie series."""
    return max(3, int(math.ceil(x + 4.0 * x ** (1.0 / 3.0) + 2.0)))


def _riccati(z: complex, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Riccati-Bessel functions psi_n(z) and xi_n(z) for n = -1 .. n_max.

    ``result[n + 1]`` holds the value for order ``n``.
    """
    orders = np.arange(-1, n_max + 1) + 0.5
    prefactor = np.sqrt(np.pi * z / 2.0)
    return prefactor * jv(orders, z), prefactor * hankel1(orders, z)


def _derivatives(values: np.ndarray, z: complex, n_max: int) -> np.ndarray:
    """psi_n'(z) = psi_{n-1}(z) - n psi_n(z)/z, for n = 0 .. n_max."""
    orders = np.arange(0, n_max + 1)
    return values[0 : n_max + 1] - orders * values[1 : n_max + 2] / z


def mie_coefficients(m: complex, x: float) -> tuple[np.ndarray, np.ndarray]:
    """Scattering coefficients a_n, b_n for n = 1 .. n_max."""
    n_max = wiscombe_terms(x)
    m = complex(m)
    mx = m * x

    psi_x, xi_x = _riccati(x, n_max)
    psi_mx, _ = _riccati(mx, n_max)
    dpsi_x = _derivatives(psi_x, x, n_max)
    dxi_x = _derivatives(xi_x, x, n_max)
    dpsi_mx = _derivatives(psi_mx, mx, n_max)

    n = np.arange(1, n_max + 1)
    psi_n_x, dpsi_n_x = psi_x[n + 1], dpsi_x[n]
    xi_n_x, dxi_n_x = xi_x[n + 1], dxi_x[n]
    psi_n_mx, dpsi_n_mx = psi_mx[n + 1], dpsi_mx[n]

    a = (m * psi_n_mx * dpsi_n_x - psi_n_x * dpsi_n_mx) / (
        m * psi_n_mx * dxi_n_x - xi_n_x * dpsi_n_mx
    )
    b = (psi_n_mx * dpsi_n_x - m * psi_n_x * dpsi_n_mx) / (
        psi_n_mx * dxi_n_x - m * xi_n_x * dpsi_n_mx
    )
    return a, b


def mie_efficiencies_reference(
    m: complex, diameter: float, wavelength: float, n_env: float = 1.0
) -> dict[str, float]:
    """Efficiencies for one homogeneous sphere, in the caller's own units.

    ``diameter`` and ``wavelength`` only need to be consistent with each other.
    """
    m_rel = complex(m) / n_env
    x = np.pi * diameter / (wavelength / n_env)
    if x <= 0.0:
        raise ValueError("diameter and wavelength must be positive")

    a, b = mie_coefficients(m_rel, x)
    n = np.arange(1, len(a) + 1)
    weight = 2.0 * n + 1.0
    prefactor = 2.0 / x**2

    q_ext = prefactor * np.sum(weight * np.real(a + b))
    q_sca = prefactor * np.sum(weight * (np.abs(a) ** 2 + np.abs(b) ** 2))

    if q_sca > 0.0:
        forward = np.sum(
            (n[:-1] * (n[:-1] + 2.0) / (n[:-1] + 1.0))
            * np.real(a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:]))
        )
        cross = np.sum((weight / (n * (n + 1.0))) * np.real(a * np.conj(b)))
        g = (4.0 / x**2) * (forward + cross) / q_sca
    else:
        g = 0.0

    return {
        "Qext": float(q_ext),
        "Qsca": float(q_sca),
        "Qabs": float(q_ext - q_sca),
        "g": float(g),
    }


__all__ = ["mie_efficiencies_reference", "mie_coefficients", "wiscombe_terms"]
