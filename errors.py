"""Error hierarchy for optcon.

Every failure mode that this library is supposed to *catch* gets its own
exception class, so that callers can distinguish "your physics is wrong" from
"your code is wrong".
"""

from __future__ import annotations


class OptConError(Exception):
    """Base class for every optcon failure."""


class UnitError(OptConError):
    """A unit name is unknown or a unit expression is malformed."""


class DimensionError(OptConError):
    """Two quantities with incompatible dimensions were combined."""


class AmplitudeOrderError(OptConError):
    """A field amplitude and a power-like quantity were combined directly.

    ``transmission`` and ``sqrt(transmission)`` are not interchangeable; this
    error is raised when the algebra proves they were mixed.
    """


class ContractViolation(OptConError):
    """A declared physical contract does not hold."""


class AdjointCheckFailure(ContractViolation):
    """A supplied adjoint/derivative disagrees with finite differences."""
