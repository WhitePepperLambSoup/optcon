"""Evidence requirements for promoting numerical results to decisions.

The gate is deliberately small.  It does not certify a model or estimate the
accuracy of arbitrary software; it records whether the evidence declared by a
specific decision is present before a result is reported as decision-ready.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .errors import EvidenceGateFailure

_REQUIREMENT_NAMES = (
    "semantic",
    "numerical",
    "independent_reference",
    "provenance",
    "scope",
)


@dataclass(frozen=True)
class EvidenceRequirements:
    """Evidence categories required by one downstream decision."""

    semantic: bool = True
    numerical: bool = True
    independent_reference: bool = False
    provenance: bool = True
    scope: bool = True


@dataclass(frozen=True)
class EvidenceNode:
    """One composable evidence artifact in a promotion graph."""

    node_id: str
    categories: frozenset[str]
    source: str
    scope: str
    parents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("scope must be a non-empty string")

        categories = frozenset(self.categories)
        unknown = categories.difference(_REQUIREMENT_NAMES)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown evidence categories: {names}")
        parents = tuple(self.parents)
        if any(not isinstance(parent, str) or not parent.strip() for parent in parents):
            raise ValueError("parents must contain non-empty string identifiers")
        if len(set(parents)) != len(parents):
            raise ValueError("parents must not contain duplicate identifiers")
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "parents", parents)


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
    graph_node_id: str | None = None
    evidence_categories: tuple[str, ...] = ()

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


class EvidenceGraph:
    """Directed acyclic graph that composes evidence categories by ancestry."""

    def __init__(self, nodes: Iterable[EvidenceNode] = ()) -> None:
        self._nodes: dict[str, EvidenceNode] = {}
        for node in nodes:
            if node.node_id in self._nodes:
                raise ValueError(f"duplicate evidence node: {node.node_id}")
            self._nodes[node.node_id] = node
        self._validate_graph()

    def add(self, node: EvidenceNode) -> None:
        """Add one node after validating parents and preserving acyclicity."""
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate evidence node: {node.node_id}")
        missing = [parent for parent in node.parents if parent not in self._nodes]
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"unknown parent evidence node(s): {names}")
        self._nodes[node.node_id] = node
        try:
            self._validate_graph()
        except Exception:
            del self._nodes[node.node_id]
            raise

    def available_categories(self, node_id: str) -> tuple[str, ...]:
        """Return the ordered category closure for a target node."""
        categories = self._category_closure(node_id)
        return tuple(name for name in _REQUIREMENT_NAMES if name in categories)

    def as_record(self, node_id: str) -> EvidenceRecord:
        """Convert a graph closure into the legacy Boolean evidence record."""
        categories = set(self.available_categories(node_id))
        return EvidenceRecord(
            semantic="semantic" in categories,
            numerical="numerical" in categories,
            independent_reference="independent_reference" in categories,
            provenance="provenance" in categories,
            scope="scope" in categories,
        )

    def missing_categories(
        self, node_id: str, requirements: EvidenceRequirements
    ) -> tuple[str, ...]:
        """Return required categories absent from a target's ancestry."""
        record = self.as_record(node_id)
        return tuple(
            name
            for name in _REQUIREMENT_NAMES
            if getattr(requirements, name) and not getattr(record, name)
        )

    def evaluate(self, node_id: str, claim: EvidenceClaim) -> EvidenceDecision:
        """Evaluate a claim from graph-derived evidence and retain its trace."""
        categories = self.available_categories(node_id)
        result = evaluate_claim(claim, self.as_record(node_id))
        return EvidenceDecision(
            claim=result.claim,
            record=result.record,
            decision_ready=result.decision_ready,
            failed_requirements=result.failed_requirements,
            graph_node_id=node_id,
            evidence_categories=categories,
        )

    def _validate_graph(self) -> None:
        for node in self._nodes.values():
            missing = [parent for parent in node.parents if parent not in self._nodes]
            if missing:
                names = ", ".join(missing)
                raise ValueError(f"unknown parent evidence node(s): {names}")
        for node_id in self._nodes:
            self._category_closure(node_id)

    def _category_closure(
        self,
        node_id: str,
        visiting: set[str] | None = None,
        visited: set[str] | None = None,
    ) -> set[str]:
        if node_id not in self._nodes:
            raise KeyError(f"unknown evidence node: {node_id}")
        active = set() if visiting is None else visiting
        complete = set() if visited is None else visited
        if node_id in active:
            raise ValueError(f"evidence graph cycle detected at: {node_id}")
        if node_id in complete:
            return set()

        active.add(node_id)
        node = self._nodes[node_id]
        categories = set(node.categories)
        for parent in node.parents:
            categories.update(self._category_closure(parent, active, complete))
        active.remove(node_id)
        complete.add(node_id)
        return categories


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
    "EvidenceGraph",
    "EvidenceGateResult",
    "EvidenceNode",
    "EvidenceRecord",
    "EvidenceRequirements",
    "evaluate_claim",
    "evaluate_evidence",
    "require_decision_ready",
]
