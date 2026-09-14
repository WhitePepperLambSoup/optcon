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


def test_adjoint_of_the_wrong_shape_is_reported_clearly():
    def forward(x):
        return np.array([x[0], x[1]])

    with pytest.raises(AdjointCheckFailure, match="shape"):
        assert_adjoint(forward, lambda y: np.zeros(5), np.array([1.0, 2.0]))


def test_assert_gradient_matches_accepts_the_true_gradient():
    x = np.array([1.0, 2.0])
    assert_gradient_matches(quadratic, quadratic_grad, x)
