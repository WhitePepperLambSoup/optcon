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
from .fox_li import (
    FoxLiMode,
    FoxLiResonator,
    fox_li_operator,
    fox_li_power_iteration,
    fresnel_number,
    solve_fox_li_modes,
)
from .nlse import (
    FiberParameters,
    Pulse,
    gaussian_pulse,
    soliton_parameters,
    soliton_pulse,
    solve_nlse,
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

__version__ = "0.1.1"

__all__ = [
    "AMPLITUDE",
    "AdjointCheckFailure",
    "AmplitudeOrderError",
    "ContractViolation",
    "Dimension",
    "DimensionError",
    "FiberParameters",
    "FoxLiMode",
    "FoxLiResonator",
    "OptConError",
    "POWER",
    "Pulse",
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
    "fox_li_operator",
    "fox_li_power_iteration",
    "fresnel_number",
    "gain_above_unity",
    "gaussian_pulse",
    "is_passive",
    "is_reciprocal",
    "is_unitary",
    "power_ratio",
    "q",
    "singular_values",
    "soliton_parameters",
    "soliton_pulse",
    "solve_fox_li_modes",
    "solve_nlse",
    "sqrt",
    "unit",
    "unitarity_error",
]
