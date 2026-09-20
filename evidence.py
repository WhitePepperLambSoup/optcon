"""Evidence requirements for promoting numerical results to decisions.

The gate is deliberately small.  It does not certify a model or estimate the
accuracy of arbitrary software; it records whether the evidence declared by a
specific decision is present before a result is reported as decision-ready.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import EvidenceGateFailure


@dataclass(frozen=True)
class EvidenceRequirements:
    """Evidence categories required by one downstream decision."""

    semantic: bool = True
    numerical: bool = True
    independent_reference: bool = False
    provenance: bool = True
    scope: bool = True


@dataclass(frozen=True)
class EvidenceClaim:
    """Auditable description of one numerical result and its decision scope.

    The metadata is intentionally required. A Boolean gate without an
    observable, evidence source, and scope can report a result as ready while
    leaving unclear what was actually promoted.
    """

    claim_id: str
    representation: str
    observable: str
    tolerance: str
    diagnostic: str
    evidence_source: str
    scope: str
    requirements: EvidenceRequirements = EvidenceRequirements()

    def __post_init__(self) -> None:
        for name in (
            "claim_id",
            "representation",
            "observable",
            "tolerance",
            "diagnostic",
            "evidence_source",
            "scope",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class EvidenceRecord:
    """Evidence actually available for one reported numerical result."""

    semantic: bool
    numerical: bool
    independent_reference: bool
    provenance: bool
    scope: bool


@dataclass(frozen=True)
class EvidenceGateResult:
    """Outcome of evaluating a result against declared evidence requirements."""

    decision_ready: bool
    failed_requirements: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceDecision:
    """Structured promotion outcome retaining the claim and evidence trace."""

    claim: EvidenceClaim
    record: EvidenceRecord
    decision_ready: bool
    failed_requirements: tuple[str, ...]

    @property
    def status(self) -> str:
        """Return the reportable workflow status for the decision."""
        return "ready" if self.decision_ready else "blocked"

    @property
    def required_categories(self) -> tuple[str, ...]:
        """Evidence categories declared by the decision."""
        return tuple(
            name for name in _REQUIREMENT_NAMES if getattr(self.claim.requirements, name)
        )

    @property
    def available_categories(self) -> tuple[str, ...]:
        """Evidence categories present in the evaluated record."""
        return tuple(name for name in _REQUIREMENT_NAMES if getattr(self.record, name))


_REQUIREMENT_NAMES = (
    "semantic",
    "numerical",
    "independent_reference",
    "provenance",
    "scope",
)


def evaluate_evidence(
    record: EvidenceRecord,
    requirements: EvidenceRequirements | None = None,
) -> EvidenceGateResult:
    """Evaluate whether ``record`` satisfies the required evidence categories."""
    required = requirements or EvidenceRequirements()
    failed = tuple(
        name
        for name in _REQUIREMENT_NAMES
        if getattr(required, name) and not getattr(record, name)
    )
    return EvidenceGateResult(decision_ready=not failed, failed_requirements=failed)


def evaluate_claim(claim: EvidenceClaim, record: EvidenceRecord) -> EvidenceDecision:
    """Evaluate a claim while retaining its audit metadata and evidence trace."""
    result = evaluate_evidence(record, claim.requirements)
    return EvidenceDecision(
        claim=claim,
        record=record,
        decision_ready=result.decision_ready,
        failed_requirements=result.failed_requirements,
    )


def require_decision_ready(
    record: EvidenceRecord,
    requirements: EvidenceRequirements | None = None,
) -> EvidenceGateResult:
    """Return a passing result or raise a diagnostic contract failure."""
    result = evaluate_evidence(record, requirements)
    if not result.decision_ready:
        raise EvidenceGateFailure(result.failed_requirements)
    return result


__all__ = [
    "EvidenceClaim",
    "EvidenceDecision",
    "EvidenceGateResult",
    "EvidenceRecord",
    "EvidenceRequirements",
    "evaluate_claim",
    "evaluate_evidence",
    "require_decision_ready",
]
