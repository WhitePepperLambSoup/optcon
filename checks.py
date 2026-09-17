"""Physical invariant contracts for optical operators.

Each ``assert_*`` turns a property that is normally carried in the author's
head - "this waveplate is lossless", "this splitter cannot amplify", "this
travelling-wave element is reciprocal" - into a checkable, composable claim.

Matrices are interpreted as Jones/transfer operators acting on field
amplitudes, so:

* ``unitary``   -> lossless (energy preserving) and polarisation preserving
* ``passive``   -> no singular value exceeds one, i.e. no net gain
* ``reciprocal``-> symmetric in the amplitude basis
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import AdjointCheckFailure, ContractViolation, DimensionError
from .quantity import Quantity

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12
DEFAULT_EPS = 1e-6
DEFAULT_GRAD_RTOL = 1e-5
DEFAULT_DOT_RTOL = 1e-6


def _as_matrix(value: Any, name: str = "matrix") -> np.ndarray:
    if isinstance(value, Quantity):
        if not value.unit.is_dimensionless:
            raise DimensionError(
                f"{name}: invariant checks apply to dimensionless operators, "
                f"got {value.unit}"
            )
        value = value.value
    matrix = np.asarray(value)
    if matrix.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D operator, got shape {matrix.shape}")
    if matrix.size == 0:
        raise ValueError(f"{name}: expected a non-empty operator, got shape {matrix.shape}")
    return matrix.astype(complex, copy=False)


def _require_square(matrix: np.ndarray, name: str) -> None:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name}: expected a square operator, got shape {matrix.shape}")


def singular_values(matrix: Any) -> np.ndarray:
    """Singular values of the operator, in descending order."""
    return np.linalg.svd(_as_matrix(matrix), compute_uv=False)


def gain_above_unity(matrix: Any) -> float:
    """How far the largest singular value exceeds unity (0 for passive)."""
    largest = float(np.max(singular_values(matrix)))
    return max(0.0, largest - 1.0)


def unitarity_error(matrix: Any) -> float:
    """``max|M^H M - I|``; zero exactly when the operator is lossless."""
    operator = _as_matrix(matrix)
    _require_square(operator, "matrix")
    product = operator.conj().T @ operator
    return float(np.max(np.abs(product - np.eye(operator.shape[0]))))


def reciprocity_error(matrix: Any) -> float:
    """``max|M - M^T|``; evaluates to zero when the operator matrix is symmetric.

    Under standard symmetric port/basis conventions (such as spatial kernel symmetry
    K(x1, x2) = K(x2, x1) in isotropic open resonators and reciprocal multi-port
    scattering matrices with identical port normalization), matrix symmetry M = M^T
    is the algebraic manifestation of electromagnetic reciprocity (Lorentz reciprocity).
    Note: Magneto-optic, non-reciprocal media (e.g. Faraday isolators) or asymmetric
    port bases break this relation.
    """
    operator = _as_matrix(matrix)
    _require_square(operator, "matrix")
    return float(np.max(np.abs(operator - operator.T)))


def is_unitary(matrix: Any, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> bool:
    operator = _as_matrix(matrix)
    if operator.shape[0] != operator.shape[1]:
        return False
    product = operator.conj().T @ operator
    return bool(np.allclose(product, np.eye(operator.shape[0]), rtol=rtol, atol=atol))


def is_passive(matrix: Any, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> bool:
    try:
        operator = _as_matrix(matrix)
        if not np.isfinite(operator).all():
            return False
        largest = float(np.max(np.linalg.svd(operator, compute_uv=False)))
    except (ValueError, np.linalg.LinAlgError):
        return False
    return bool(max(0.0, largest - 1.0) <= rtol + atol)


def is_reciprocal(matrix: Any, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> bool:
    operator = _as_matrix(matrix)
    if operator.shape[0] != operator.shape[1]:
        return False
    return bool(np.allclose(operator, operator.T, rtol=rtol, atol=atol))


def assert_unitary(
    matrix: Any,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    name: str = "operator",
) -> None:
    """Declare an operator lossless; raise if it is not."""
    if is_unitary(matrix, rtol=rtol, atol=atol):
        return
    raise ContractViolation(
        f"{name}: declared unitary (lossless) but max|M^H M - I| = "
        f"{unitarity_error(matrix):.3e} exceeds tolerance {rtol:g}/{atol:g}"
    )


def assert_passive(
    matrix: Any,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    name: str = "operator",
) -> None:
    """Declare an operator gain-free; raise if it amplifies."""
    operator = _as_matrix(matrix, name)
    if not np.isfinite(operator).all():
        raise ContractViolation(
            f"{name}: declared passive but contains non-finite matrix entries"
        )
    if is_passive(matrix, rtol=rtol, atol=atol):
        return
    try:
        excess = gain_above_unity(operator)
    except np.linalg.LinAlgError as error:
        raise ContractViolation(
            f"{name}: declared passive but its singular values could not be computed"
        ) from error
    raise ContractViolation(
        f"{name}: declared passive but the largest singular value exceeds unity "
        f"by {excess:.3e}"
    )


def assert_reciprocal(
    matrix: Any,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    name: str = "operator",
) -> None:
    """Declare a travelling-wave operator reciprocal; raise if it is not."""
    if is_reciprocal(matrix, rtol=rtol, atol=atol):
        return
    raise ContractViolation(
        f"{name}: declared reciprocal but max|M - M^T| = "
        f"{reciprocity_error(matrix):.3e} exceeds tolerance {rtol:g}/{atol:g}"
    )


# ---------------------------------------------------------------------------
# Derivative verification
# ---------------------------------------------------------------------------


def _as_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name}: expected a 1-D vector, got shape {array.shape}")
    return array


def _validate_eps(eps: float) -> float:
    try:
        step = float(eps)
    except (TypeError, ValueError) as error:
        raise ValueError("eps must be finite and strictly positive") from error
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("eps must be finite and strictly positive")
    return step


def finite_difference_gradient(
    function, x: Any, eps: float = DEFAULT_EPS
) -> np.ndarray:
    """Central-difference gradient of a scalar function."""
    point = np.asarray(_as_vector(x, "x"), dtype=np.result_type(np.asarray(x).dtype, float))
    step_size = _validate_eps(eps)
    gradient = np.zeros(point.shape, dtype=np.result_type(point.dtype, float))
    for index in range(point.size):
        step = np.zeros_like(point, dtype=point.dtype)
        step[index] = step_size
        gradient[index] = (
            function(point + step) - function(point - step)
        ) / (2.0 * step_size)
    return gradient


def finite_difference_jacobian(function, x: Any, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Central-difference Jacobian of a vector-valued function."""
    point = np.asarray(_as_vector(x, "x"), dtype=np.result_type(np.asarray(x).dtype, float))
    step_size = _validate_eps(eps)
    columns = []
    for index in range(point.size):
        step = np.zeros_like(point, dtype=point.dtype)
        step[index] = step_size
        columns.append(
            (np.asarray(function(point + step)) - np.asarray(function(point - step)))
            / (2.0 * step_size)
        )
    return np.column_stack(columns)


def check_gradient(
    function,
    gradient_function,
    x: Any,
    eps: float = DEFAULT_EPS,
    rtol: float = DEFAULT_GRAD_RTOL,
    atol: float = 1e-12,
) -> dict:
    """Compare a supplied gradient against central differences.

    Returns a report rather than raising, so callers can log the residual.
    """
    point = _as_vector(x, "x")
    numerical = finite_difference_gradient(function, point, eps)
    analytic = _as_vector(gradient_function(point), "gradient")
    if analytic.shape != numerical.shape:
        raise ValueError(
            f"gradient shape {analytic.shape} does not match the input shape {numerical.shape}"
        )
    denominator = np.maximum(np.maximum(np.abs(analytic), np.abs(numerical)), atol)
    relative = np.abs(analytic - numerical) / denominator
    worst = int(np.argmax(relative))
    return {
        "ok": bool(relative[worst] <= rtol),
        "max_rel_error": float(relative[worst]),
        "worst_index": worst,
        "rtol": rtol,
        "analytic": analytic,
        "numerical": numerical,
    }


def assert_gradient_matches(
    function,
    gradient_function,
    x: Any,
    eps: float = DEFAULT_EPS,
    rtol: float = DEFAULT_GRAD_RTOL,
    name: str = "gradient",
) -> None:
    """Declare a gradient correct; raise if it is not the real derivative."""
    report = check_gradient(function, gradient_function, x, eps=eps, rtol=rtol)
    if report["ok"]:
        return
    worst = report["worst_index"]
    raise AdjointCheckFailure(
        f"{name}: derivative disagrees with finite differences; max relative error "
        f"{report['max_rel_error']:.3e} at index {worst} exceeds rtol {rtol:g} "
        f"(supplied {report['analytic'][worst]:.6g}, numerical {report['numerical'][worst]:.6g})"
    )


def dot_test(
    forward,
    adjoint,
    x: Any,
    seed: int = 0,
    eps: float = DEFAULT_EPS,
    rtol: float = DEFAULT_DOT_RTOL,
) -> dict:
    """The classic complex inner-product adjoint check.

    ``Jv`` comes from central differences of ``forward``; ``J^H w`` comes from
    the supplied ``adjoint``.  The two inner products must agree for random
    probes ``v`` and ``w``.
    """
    point = _as_vector(x, "x")
    step_size = _validate_eps(eps)
    output = _as_vector(forward(point), "forward(x)")
    generator = np.random.default_rng(seed)
    def random_probe(size: int, template: np.ndarray) -> np.ndarray:
        real = generator.standard_normal(size)
        if np.iscomplexobj(template):
            return real + 1.0j * generator.standard_normal(size)
        return real

    probe_in = random_probe(point.size, point)
    probe_out = random_probe(output.size, output)
    output_plus = _as_vector(
        forward(point + step_size * probe_in), "forward(x + eps v)"
    )
    output_minus = _as_vector(
        forward(point - step_size * probe_in), "forward(x - eps v)"
    )
    if output_plus.shape != output.shape or output_minus.shape != output.shape:
        raise AdjointCheckFailure(
            "forward output shape changed under the finite-difference perturbation: "
            f"forward(x)={output.shape}, forward(x + eps v)={output_plus.shape}, "
            f"forward(x - eps v)={output_minus.shape}"
        )
    forward_probe = (output_plus - output_minus) / (2.0 * step_size)
    adjoint_probe = adjoint(probe_out)
    try:
        adjoint_probe = _as_vector(adjoint_probe, "adjoint(w)")
    except ValueError as error:
        raise AdjointCheckFailure(f"adjoint returned the wrong shape: {error}") from None
    if adjoint_probe.shape != point.shape:
        raise AdjointCheckFailure(
            f"adjoint(w) has shape {adjoint_probe.shape} but must have shape {point.shape}"
        )
    left = np.vdot(forward_probe, probe_out)
    right = np.vdot(probe_in, adjoint_probe)
    scale = max(abs(left), abs(right), 1e-300)
    return {
        "ok": bool(abs(left - right) / scale <= rtol),
        "left": left,
        "right": right,
        "rel_error": float(abs(left - right) / scale),
        "rtol": rtol,
    }


def assert_adjoint(
    forward,
    adjoint,
    x: Any,
    seed: int = 0,
    eps: float = DEFAULT_EPS,
    rtol: float = DEFAULT_DOT_RTOL,
    name: str = "operator",
) -> None:
    """Declare ``adjoint`` the adjoint of ``forward``; raise if it is not."""
    report = dot_test(forward, adjoint, x, seed=seed, eps=eps, rtol=rtol)
    if report["ok"]:
        return
    raise AdjointCheckFailure(
        f"{name}: adjoint test failed with relative error {report['rel_error']:.3e} "
        f"exceeding rtol {rtol:g} (<Jv, w> = {report['left']:.6g}, "
        f"<v, J^H w> = {report['right']:.6g})"
    )
