"""Verification that a supplied derivative really is the derivative.

The motivating failure is a training loop whose gradient looks plausible
(loss goes down, curves are smooth) while the reverse pass is not the adjoint
of the forward pass.  Nothing in the usual stack notices; these checks do.
"""

import numpy as np
import pytest

from optcon import AdjointCheckFailure
from optcon.checks import (
    assert_adjoint,
    assert_gradient_matches,
    check_gradient,
    dot_test,
    finite_difference_gradient,
    finite_difference_jacobian,
)


def quadratic(x):
    return float(np.sum(x**2))


def quadratic_grad(x):
    return 2.0 * x


def test_finite_difference_gradient_matches_analytic_gradient():
    x = np.array([0.3, -1.2, 2.5])
    assert finite_difference_gradient(quadratic, x) == pytest.approx(quadratic_grad(x))


def test_finite_difference_jacobian_of_a_linear_map_is_the_map():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    jacobian = finite_difference_jacobian(lambda x: matrix @ x, np.zeros(2))
    assert jacobian == pytest.approx(matrix)


def test_finite_difference_helpers_promote_integer_points_to_floating_dtype():
    point = np.array([1, 2], dtype=int)

    gradient = finite_difference_gradient(lambda x: float(np.sum(x**2)), point)
    jacobian = finite_difference_jacobian(lambda x: 2.0 * x, point)

    assert gradient.dtype.kind == "f"
    assert gradient == pytest.approx([2.0, 4.0])
    assert jacobian.dtype.kind == "f"
    assert jacobian == pytest.approx(2.0 * np.eye(2))


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
def test_finite_difference_helpers_reject_invalid_step_sizes(eps):
    with pytest.raises(ValueError, match="eps must be finite and strictly positive"):
        finite_difference_gradient(quadratic, np.array([1.0, 2.0]), eps=eps)


def test_correct_gradient_passes():
    x = np.array([0.3, -1.2, 2.5])
    report = check_gradient(quadratic, quadratic_grad, x)
    assert report["ok"] is True
    assert report["max_rel_error"] < 1e-5


def test_gradient_scaled_by_two_is_rejected():
    x = np.array([0.3, -1.2, 2.5])
    with pytest.raises(AdjointCheckFailure):
        assert_gradient_matches(quadratic, lambda v: 4.0 * v, x, name="actor gradient")


def test_check_gradient_reports_the_worst_component():
    x = np.array([0.0, 0.0, 0.0])
    report = check_gradient(quadratic, lambda v: np.array([0.0, 0.0, 5.0]), x)
    assert report["ok"] is False
    assert report["worst_index"] == 2


def test_dot_test_accepts_a_true_adjoint_pair():
    matrix = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]])

    def forward(x):
        return matrix @ x

    def adjoint(y):
        return matrix.T @ y

    assert dot_test(forward, adjoint, np.array([0.5, -0.5, 1.0]))["ok"] is True


def test_dot_test_rejects_a_forward_map_masquerading_as_its_own_adjoint():
    matrix = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]])
    incomplete = matrix.copy()
    incomplete[0, 1] = 0.0  # reverse pass silently drops one coupling term

    def forward(x):
        return matrix @ x

    def wrong_adjoint(y):
        return incomplete.T @ y

    with pytest.raises(AdjointCheckFailure):
        assert_adjoint(forward, wrong_adjoint, np.array([0.5, -0.5, 1.0]), name="propagator")


def test_dot_test_uses_the_complex_hermitian_inner_product():
    matrix = np.array([[1.0 + 2.0j, 2.0 - 1.0j], [0.5j, 1.0 - 0.25j]])

    def forward(x):
        return matrix @ x

    def adjoint(y):
        return matrix.conj().T @ y

    report = dot_test(forward, adjoint, np.array([0.5 + 0.2j, -0.5 + 0.1j]))
    assert report["ok"] is True


def test_dot_test_uses_the_real_inner_product_for_real_parameters_and_complex_fields():
    matrix = np.array([[1.0 + 2.0j], [-0.5 + 0.25j]])

    def forward(x):
        return matrix @ x

    def real_adjoint(y):
        return np.real(matrix.conj().T @ y)

    report = dot_test(forward, real_adjoint, np.array([0.3]), seed=7)

    assert report["ok"] is True
    assert report["inner_product"] == "real"
    assert np.isrealobj(report["left"])
    assert np.isrealobj(report["right"])


def test_dot_test_rejects_a_complex_cotangent_for_a_real_parameter_space():
    matrix = np.array([[1.0 + 2.0j], [-0.5 + 0.25j]])

    def forward(x):
        return matrix @ x

    def complex_adjoint(y):
        return matrix.conj().T @ y

    report = dot_test(forward, complex_adjoint, np.array([0.3]), seed=7)

    assert report["ok"] is False
    assert report["adjoint_domain_error"] > 0.0


def test_assert_adjoint_names_a_real_domain_cotangent_error():
    matrix = np.array([[1.0 + 2.0j], [-0.5 + 0.25j]])

    def forward(x):
        return matrix @ x

    def complex_adjoint(y):
        return matrix.conj().T @ y

    with pytest.raises(
        AdjointCheckFailure,
        match="complex cotangent.*real parameter space",
    ):
        assert_adjoint(
            forward,
            complex_adjoint,
            np.array([0.3]),
            seed=7,
            name="real-parameter adjoint",
        )


def test_assert_adjoint_reports_the_real_dual_pairing_on_pairing_failure():
    matrix = np.array([[1.0 + 2.0j], [-0.5 + 0.25j]])

    def forward(x):
        return matrix @ x

    def incorrectly_scaled_real_adjoint(y):
        return 0.5 * np.real(matrix.conj().T @ y)

    with pytest.raises(
        AdjointCheckFailure,
        match=r"real adjoint test failed.*Re<Jv, w>.*J\*_R",
    ):
        assert_adjoint(
            forward,
            incorrectly_scaled_real_adjoint,
            np.array([0.3]),
            seed=7,
            name="real-parameter adjoint",
        )


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
def test_dot_test_rejects_invalid_step_sizes(eps):
    def forward(x):
        return 2.0 * x

    def adjoint(w):
        return 2.0 * w

    with pytest.raises(ValueError, match="eps must be finite and strictly positive"):
        dot_test(forward, adjoint, np.array([1.0]), eps=eps)


def test_adjoint_of_the_wrong_shape_is_reported_clearly():
    def forward(x):
        return np.array([x[0], x[1]])

    with pytest.raises(AdjointCheckFailure, match="shape"):
        assert_adjoint(forward, lambda y: np.zeros(5), np.array([1.0, 2.0]))


def test_dot_test_rejects_forward_output_shape_changes_under_perturbation():
    def forward(x):
        if x[0] > 0.0:
            return np.array([x[0]])
        return np.array([x[0], x[0]])

    def adjoint(w):
        return np.array([np.sum(w)])

    with pytest.raises(AdjointCheckFailure, match="forward output shape changed"):
        dot_test(forward, adjoint, np.array([0.0]), seed=0, eps=1e-6)


def test_assert_gradient_matches_accepts_the_true_gradient():
    x = np.array([1.0, 2.0])
    assert_gradient_matches(quadratic, quadratic_grad, x)
