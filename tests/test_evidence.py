"""Tests for the decision-evidence promotion gate."""

from __future__ import annotations

from itertools import product

import pytest

from optcon import (
    EvidenceClaim,
    EvidenceGateFailure,
    EvidenceGraph,
    EvidenceNode,
    EvidenceRecord,
    EvidenceRequirements,
    evaluate_claim,
    evaluate_evidence,
    require_decision_ready,
    threshold_stability,
)
from optcon.benchmarks.evidence_ablation import (
    EVIDENCE_CATEGORIES,
    run_evidence_requirement_ablation,
)


def test_default_requirements_accept_complete_local_evidence():
    record = EvidenceRecord(
        semantic=True,
        numerical=True,
        independent_reference=False,
        provenance=True,
        scope=True,
    )

    result = evaluate_evidence(record)

    assert result.decision_ready is True
    assert result.failed_requirements == ()


def test_required_independent_reference_blocks_an_unadjudicated_result():
    record = EvidenceRecord(
        semantic=True,
        numerical=True,
        independent_reference=False,
        provenance=True,
        scope=True,
    )
    requirements = EvidenceRequirements(independent_reference=True)

    result = evaluate_evidence(record, requirements)

    assert result.decision_ready is False
    assert result.failed_requirements == ("independent_reference",)


def test_optional_requirements_are_not_reported_as_failures():
    record = EvidenceRecord(
        semantic=True,
        numerical=True,
        independent_reference=False,
        provenance=True,
        scope=True,
    )
    requirements = EvidenceRequirements(
        semantic=True,
        numerical=True,
        independent_reference=True,
        provenance=False,
        scope=False,
    )

    result = evaluate_evidence(record, requirements)

    assert result.failed_requirements == ("independent_reference",)


def test_require_decision_ready_returns_the_result_when_all_requirements_pass():
    record = EvidenceRecord(
        semantic=True,
        numerical=True,
        independent_reference=True,
        provenance=True,
        scope=True,
    )

    result = require_decision_ready(record, EvidenceRequirements(independent_reference=True))

    assert result.decision_ready is True


def test_require_decision_ready_exposes_failed_requirements():
    record = EvidenceRecord(
        semantic=True,
        numerical=False,
        independent_reference=False,
        provenance=True,
        scope=False,
    )

    with pytest.raises(EvidenceGateFailure) as caught:
        require_decision_ready(record)

    assert caught.value.failed_requirements == ("numerical", "scope")


def test_requirement_ablation_releases_only_the_blocking_category():
    rows = run_evidence_requirement_ablation()

    assert len(rows) == 3 * len(EVIDENCE_CATEGORIES)
    assert all(row.full_decision_ready is False for row in rows)
    released = {
        (row.case_id, row.removed_requirement)
        for row in rows
        if row.released_by_removal
    }
    assert released == {
        ("fault-power-used-as-field", "semantic"),
        ("vacuum-default-mismatch", "independent_reference"),
        ("omitted-terminal-high", "independent_reference"),
    }


def test_evidence_gate_is_monotone_in_available_evidence():
    """Adding evidence cannot turn a ready decision into a blocked one."""
    for required_values in product((False, True), repeat=5):
        requirements = EvidenceRequirements(*required_values)
        for lower_values in product((False, True), repeat=5):
            lower = EvidenceRecord(*lower_values)
            lower_result = evaluate_evidence(lower, requirements).decision_ready
            for upper_values in product((False, True), repeat=5):
                if not all(
                    not lower_value or upper_value
                    for lower_value, upper_value in zip(
                        lower_values, upper_values, strict=True
                    )
                ):
                    continue
                upper_result = evaluate_evidence(
                    EvidenceRecord(*upper_values), requirements
                ).decision_ready
                assert not lower_result or upper_result


def test_evidence_gate_is_monotone_under_requirement_weakening():
    """Removing requirements cannot make a ready decision fail."""
    for record_values in product((False, True), repeat=5):
        record = EvidenceRecord(*record_values)
        for stronger_values in product((False, True), repeat=5):
            stronger = EvidenceRequirements(*stronger_values)
            stronger_result = evaluate_evidence(record, stronger).decision_ready
            for weaker_values in product((False, True), repeat=5):
                if not all(
                    not weaker_value or stronger_value
                    for weaker_value, stronger_value in zip(
                        weaker_values, stronger_values, strict=True
                    )
                ):
                    continue
                weaker_result = evaluate_evidence(
                    record, EvidenceRequirements(*weaker_values)
                ).decision_ready
                assert not stronger_result or weaker_result


def test_missing_required_evidence_is_not_compensated_by_other_categories():
    """A missing required category remains blocking regardless of other evidence."""
    for required_values in product((False, True), repeat=5):
        requirements = EvidenceRequirements(*required_values)
        for record_values in product((False, True), repeat=5):
            record = EvidenceRecord(*record_values)
            for index, required in enumerate(required_values):
                if required and not record_values[index]:
                    assert not evaluate_evidence(record, requirements).decision_ready


def test_claim_evaluation_returns_an_auditable_promotion_trace():
    claim = EvidenceClaim(
        claim_id="mie-water-threshold",
        representation="relative refractive index and in-medium wavelength",
        observable="Qext threshold diameter",
        tolerance="maximum relative reference error <= 1e-6",
        diagnostic="matched same-tree Mie series",
        evidence_source="author-constructed reference implementation",
        scope="homogeneous sphere in a declared water medium",
        requirements=EvidenceRequirements(independent_reference=True),
    )
    record = EvidenceRecord(
        semantic=True,
        numerical=True,
        independent_reference=False,
        provenance=True,
        scope=True,
    )

    result = evaluate_claim(claim, record)

    assert result.claim is claim
    assert result.record is record
    assert result.status == "blocked"
    assert result.decision_ready is False
    assert result.failed_requirements == ("independent_reference",)
    assert result.required_categories == (
        "semantic",
        "numerical",
        "independent_reference",
        "provenance",
        "scope",
    )
    assert result.available_categories == (
        "semantic",
        "numerical",
        "provenance",
        "scope",
    )


def test_claim_metadata_must_be_explicit():
    with pytest.raises(ValueError, match="claim_id"):
        EvidenceClaim(
            claim_id="",
            representation="field",
            observable="power",
            tolerance="exact",
            diagnostic="contract",
            evidence_source="local",
            scope="test",
        )


def test_evidence_graph_inherits_categories_from_parents():
    graph = EvidenceGraph(
        [
            EvidenceNode("semantic", frozenset({"semantic"}), "unit test", "field"),
            EvidenceNode(
                "reference",
                frozenset({"independent_reference"}),
                "TMM",
                "stack",
            ),
            EvidenceNode(
                "result",
                frozenset({"numerical"}),
                "solver",
                "stack",
                ("semantic", "reference"),
            ),
        ]
    )

    assert graph.available_categories("result") == (
        "semantic",
        "numerical",
        "independent_reference",
    )


def test_evidence_graph_reports_missing_categories_and_compatibility_record():
    graph = EvidenceGraph(
        [EvidenceNode("result", frozenset({"semantic", "numerical"}), "solver", "beam")]
    )
    requirements = EvidenceRequirements(
        independent_reference=True,
        provenance=False,
        scope=False,
    )

    assert graph.missing_categories("result", requirements) == (
        "independent_reference",
    )
    assert graph.as_record("result") == EvidenceRecord(
        semantic=True,
        numerical=True,
        independent_reference=False,
        provenance=False,
        scope=False,
    )


def test_evidence_graph_evaluation_retains_graph_trace():
    graph = EvidenceGraph(
        [EvidenceNode("result", frozenset({"semantic", "numerical"}), "solver", "beam")]
    )
    claim = EvidenceClaim(
        claim_id="beam-radius",
        representation="sampled field",
        observable="second-moment radius",
        tolerance="relative error <= 1e-6",
        diagnostic="analytic Gaussian reference",
        evidence_source="closed form",
        scope="paraxial Gaussian beam",
        requirements=EvidenceRequirements(
            semantic=True,
            numerical=True,
            independent_reference=False,
            provenance=False,
            scope=False,
        ),
    )

    result = graph.evaluate("result", claim)

    assert result.decision_ready is True
    assert result.graph_node_id == "result"
    assert result.evidence_categories == ("semantic", "numerical")


def test_evidence_graph_rejects_duplicate_and_unknown_parent_nodes():
    graph = EvidenceGraph()
    graph.add(EvidenceNode("result", frozenset(), "solver", "beam"))

    with pytest.raises(ValueError, match="duplicate"):
        graph.add(EvidenceNode("result", frozenset(), "solver", "beam"))
    with pytest.raises(ValueError, match="unknown parent"):
        graph.add(EvidenceNode("child", frozenset(), "solver", "beam", ("missing",)))


def test_evidence_graph_rejects_cycles_during_initial_validation():
    with pytest.raises(ValueError, match="cycle"):
        EvidenceGraph(
            [
                EvidenceNode("a", frozenset({"semantic"}), "source", "scope", ("b",)),
                EvidenceNode("b", frozenset({"numerical"}), "source", "scope", ("a",)),
            ]
        )


def test_evidence_graph_closure_is_monotone_when_evidence_is_added():
    graph = EvidenceGraph(
        [EvidenceNode("result", frozenset({"semantic"}), "solver", "beam")]
    )
    before = set(graph.available_categories("result"))
    graph.add(EvidenceNode("reference", frozenset({"independent_reference"}), "TMM", "beam"))
    graph.add(EvidenceNode("checked", frozenset(), "wrapper", "beam", ("result", "reference")))

    after = set(graph.available_categories("checked"))

    assert before <= after
    assert "independent_reference" in after


def test_claim_with_unstable_result_is_not_decision_ready():
    claim = EvidenceClaim(
        "near-boundary",
        "scalar",
        "threshold",
        "margin > 0",
        "test",
        "test",
        "demo",
    )
    record = EvidenceRecord(True, True, True, True, True)
    stability = threshold_stability(1.0, 1.0, reference_error=0.01)

    result = evaluate_claim(claim, record, stability)

    assert result.decision_ready is False
    assert result.failed_requirements == ()
    assert result.stability is stability
    assert result.blocking_reasons == ("stability",)
