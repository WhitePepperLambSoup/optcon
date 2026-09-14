"""Time and memory baseline for the operations that actually cost something.

The library is pure NumPy, so its cost lives in three places: FFT propagation,
the modal basis, and anything that materialises a full grid of modes or
shifts.  This script measures each one so that an optimisation can be
justified by a number rather than by a feeling.

Run:  python -m optcon.benchmarks.bench
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np

from optcon import q
from optcon.aberrations import wavefront_pv
from optcon.modes import decompose, hermite_gauss, reconstruct
from optcon.mtf import mtf_from_psf
from optcon.propagation import Field, gaussian_field, propagate, second_moment_radius
from optcon.vector_fields import from_scalar, propagate_vector

SAMPLES = 512
WAIST_UM = 50.0
LAMBDA_NM = 633.0
EXTENT_UM = 1600.0


def measure(operation: Callable[[], Any], repeats: int = 3) -> tuple[float, float]:
    """Return (best seconds, peak MiB) for one operation."""
    gc.collect()
    best = float("inf")
    peak = 0
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        operation()
        elapsed = time.perf_counter() - start
        _, high = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        best = min(best, elapsed)
        peak = max(peak, high)
    return best, peak / 1024**2


def build_field(samples: int = SAMPLES) -> Field:
    return gaussian_field(
        waist=q(WAIST_UM, "um"),
        wavelength=q(LAMBDA_NM, "nm"),
        samples=samples,
        extent=q(EXTENT_UM, "um"),
    )


def main() -> int:
    field = build_field()
    vector = from_scalar(field)
    distance = q(5.0, "mm")
    axis = (np.arange(SAMPLES) - SAMPLES // 2) * (EXTENT_UM / SAMPLES)
    grid_x, grid_y = np.meshgrid(axis, axis)
    radius = np.hypot(grid_x, grid_y)
    psf = np.exp(-(radius**2) / (2.0 * 20.0**2))

    cases: list[tuple[str, Callable[[], Any]]] = [
        ("field construction 512^2", lambda: build_field()),
        ("propagate 512^2 (angular spectrum)", lambda: propagate(field, distance)),
        (
            "propagate 512^2 (fresnel)",
            lambda: propagate(field, distance, method="fresnel"),
        ),
        (
            "vector propagate 512^2 (both components)",
            lambda: propagate_vector(vector, distance),
        ),
        ("second moment radius 512^2", lambda: second_moment_radius(field)),
        (
            "decompose 512^2 to order 3",
            lambda: decompose(field, waist=q(WAIST_UM, "um"), max_order=3),
        ),
        (
            "hermite_gauss 512^2 single mode",
            lambda: hermite_gauss(grid_x, grid_y, 2, 1, q(WAIST_UM, "um")),
        ),
            (
                "mtf from a 512^2 psf",
                lambda: mtf_from_psf(psf, q(EXTENT_UM / SAMPLES, "um")),
            ),
    ]
    coefficients = decompose(field, waist=q(WAIST_UM, "um"), max_order=3)
    cases.extend(
        [
            (
                "reconstruct 512^2 from order 3",
                lambda: reconstruct(
                    coefficients, field, waist=q(WAIST_UM, "um")
                ),
            ),
            (
                "wavefront pv over 11 Zernike modes",
                lambda: wavefront_pv([0.0] * 10 + [25.0], q(633.0, "nm")),
            ),
        ]
    )

    header = f"{'operation':44s} {'best s':>9s} {'peak MiB':>9s}"
    print(header)
    print("-" * len(header))
    for name, operation in cases:
        seconds, peak = measure(operation)
        print(f"{name:44s} {seconds:9.4f} {peak:9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
