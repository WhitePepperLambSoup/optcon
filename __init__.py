"""optcon - dimensional, contract-checked physical computation.

The short version of the design: physical quantities carry dimensions *and*
provenance (is this a field amplitude or a power?), and physics invariants are
declared and checked rather than remembered.
"""

from .checks import (
    assert_adjoint,
    assert_gradient_matches,
    assert_passive,
    assert_reciprocal,
    assert_unitary,
    check_gradient,
    dot_test,
    finite_difference_gradient,
    finite_difference_jacobian,
    gain_above_unity,
    is_passive,
    is_reciprocal,
    is_unitary,
    singular_values,
    unitarity_error,
)
from .errors import (
    AdjointCheckFailure,
    AmplitudeOrderError,
    ContractViolation,
    DimensionError,
    OptConError,
    UnitError,
)
from .quantity import (
    AMPLITUDE,
    POWER,
    RATIO,
    Quantity,
    amplitude_ratio,
    dimensionless,
    power_ratio,
    q,
)
from .quantity import sqrt as sqrt
from .units import Dimension, Unit, unit

__version__ = "0.1.0"

__all__ = [
    "AMPLITUDE",
    "AdjointCheckFailure",
    "AmplitudeOrderError",
    "ContractViolation",
    "Dimension",
    "DimensionError",
    "OptConError",
    "POWER",
    "Quantity",
    "RATIO",
    "Unit",
    "UnitError",
    "amplitude_ratio",
    "assert_adjoint",
    "assert_gradient_matches",
    "assert_passive",
    "assert_reciprocal",
    "assert_unitary",
    "check_gradient",
    "dimensionless",
    "dot_test",
    "finite_difference_gradient",
    "finite_difference_jacobian",
    "gain_above_unity",
    "is_passive",
    "is_reciprocal",
    "is_unitary",
    "power_ratio",
    "q",
    "singular_values",
    "sqrt",
    "unit",
    "unitarity_error",
]
