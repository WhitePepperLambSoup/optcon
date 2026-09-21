"""Independent robustness benchmark for compositional promotion decisions.

The manifest is separate from ``heldout_mutations``.  It focuses on the new
evidence-graph and decision-stability semantics across four optical decision
families, rather than estimating sensitivity for arbitrary software.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from optcon import (
    EvidenceClaim,
    EvidenceGraph,
    EvidenceNode,
    EvidenceRequirements,
    choice_stability,
    threshold_stability,
)

DEFAULT_CASE_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "optcon-artifacts"
    / "benchmarks"
    / "compositional_protocol_cases.csv"
)
DEFAULT_SUMMARY_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "optcon-artifacts"
    / "benchmarks"
    / "compositional_protocol_summary.csv"
)

_CATEGORIES = (
    "semantic",
    "numerical",
    "independent_reference",
    "provenance",
    "scope",
)


@dataclass(frozen=True)
class ProtocolCaseDefinition:
    """One frozen control or fault in the compositional protocol suite."""

    case_id: str
    family: str
    description: str
    is_fault: bool
    decision_kind: str
    value: float = 0.0
    boundary: float = 0.0
    direction: str = "at_least"
    numerical_error: float = 0.0
    reference_error: float = 0.0
    parameter_error: float = 0.0
    choice_values: tuple[tuple[str, float], ...] = ()
    choice_error_budget: float = 0.0
    maximize: bool = True
    missing_category: str | None = None


@dataclass(frozen=True)
class ProtocolCaseResult:
    """Auditable outcome for one protocol case."""

    case_id: str
    family: str
    is_fault: bool
    decision_ready: bool
    detected: bool
    stable: bool
    margin: float
    blocking_reasons: tuple[str, ...]
    manifest_hash: str
    elapsed_seconds: float
    expected_outcome_met: bool


@dataclass(frozen=True)
class ProtocolSummary:
    """Aggregate detection and promotion metrics for the protocol suite."""

    total_cases: int
    faults: int
    controls: int
    detected_faults: int
    accepted_controls: int
    faulty_decision_ready: int
    fault_detection_rate: float
    control_acceptance_rate: float
    incorrect_promotion_rate: float
    manifest_hash: str


def _threshold(
    case_id: str,
    family: str,
    description: str,
    is_fault: bool,
    value: float,
    boundary: float,
    *,
    direction: str = "at_least",
    numerical_error: float = 0.0,
    reference_error: float = 0.0,
    parameter_error: float = 0.0,
    missing_category: str | None = None,
) -> ProtocolCaseDefinition:
    return ProtocolCaseDefinition(
        case_id=case_id,
        family=family,
        description=description,
        is_fault=is_fault,
        decision_kind="threshold",
        value=value,
        boundary=boundary,
        direction=direction,
        numerical_error=numerical_error,
        reference_error=reference_error,
        parameter_error=parameter_error,
        missing_category=missing_category,
    )


def _choice(
    case_id: str,
    family: str,
    description: str,
    is_fault: bool,
    values: tuple[tuple[str, float], ...],
    *,
    error_budget: float = 0.0,
    maximize: bool = True,
    missing_category: str | None = None,
) -> ProtocolCaseDefinition:
    return ProtocolCaseDefinition(
        case_id=case_id,
        family=family,
        description=description,
        is_fault=is_fault,
        decision_kind="choice",
        choice_values=values,
        choice_error_budget=error_budget,
        maximize=maximize,
        missing_category=missing_category,
    )


PROTOCOL_MANIFEST: tuple[ProtocolCaseDefinition, ...] = (
    _threshold(
        "mie-control-vacuum",
        "mie",
        "Matched vacuum extinction threshold",
        False,
        0.92,
        0.80,
        reference_error=0.02,
    ),
    _threshold(
        "mie-control-air",
        "mie",
        "Matched air extinction threshold",
        False,
        0.74,
        0.60,
        reference_error=0.03,
    ),
    _threshold(
        "mie-control-medium",
        "mie",
        "Declared liquid-medium threshold",
        False,
        0.68,
        0.50,
        parameter_error=0.04,
    ),
    _threshold(
        "mie-fault-missing-reference",
        "mie",
        "Medium convention has no matched reference",
        True,
        0.92,
        0.80,
        missing_category="independent_reference",
    ),
    _threshold(
        "mie-fault-boundary-uncertain",
        "mie",
        "Extinction threshold is inside the reference budget",
        True,
        0.81,
        0.80,
        reference_error=0.02,
    ),
    _threshold(
        "mie-fault-wrong-scope",
        "mie",
        "Particle result is reported outside its declared medium",
        True,
        0.92,
        0.80,
        missing_category="scope",
    ),
    _choice(
        "thinfilm-control-six-pair",
        "thinfilm",
        "Six-pair coating is separated from the runner-up",
        False,
        (("six-pair", 0.96), ("seven-pair", 0.93)),
        error_budget=0.01,
    ),
    _choice(
        "thinfilm-control-eight-pair",
        "thinfilm",
        "Eight-pair coating is separated from the runner-up",
        False,
        (("eight-pair", 0.985), ("nine-pair", 0.94)),
        error_budget=0.02,
    ),
    _choice(
        "thinfilm-control-low-loss",
        "thinfilm",
        "Low-loss stack choice has a positive gap",
        False,
        (("design-a", 0.88), ("design-b", 0.80)),
        error_budget=0.03,
    ),
    _choice(
        "thinfilm-fault-missing-reference",
        "thinfilm",
        "Layer sequence lacks transfer-matrix evidence",
        True,
        (("six-pair", 0.96), ("seven-pair", 0.93)),
        error_budget=0.01,
        missing_category="independent_reference",
    ),
    _choice(
        "thinfilm-fault-tied-choice",
        "thinfilm",
        "Two pair-count choices are indistinguishable within error",
        True,
        (("six-pair", 0.96), ("seven-pair", 0.95)),
        error_budget=0.02,
    ),
    _choice(
        "thinfilm-fault-unproven-scope",
        "thinfilm",
        "Coating result omits its layer-model scope",
        True,
        (("six-pair", 0.96), ("seven-pair", 0.93)),
        error_budget=0.01,
        missing_category="scope",
    ),
    _threshold(
        "beam-control-small-radius",
        "beam",
        "Beam radius stays below the aperture limit",
        False,
        0.020,
        0.100,
        direction="at_most",
        numerical_error=0.005,
    ),
    _threshold(
        "beam-control-mid-radius",
        "beam",
        "Propagated radius remains inside the aperture",
        False,
        0.040,
        0.100,
        direction="at_most",
        numerical_error=0.010,
    ),
    _threshold(
        "beam-control-wide-aperture",
        "beam",
        "Wide aperture leaves a stable radius margin",
        False,
        0.080,
        0.200,
        direction="at_most",
        numerical_error=0.020,
    ),
    _threshold(
        "beam-fault-missing-numerics",
        "beam",
        "Radius result has no refinement evidence",
        True,
        0.020,
        0.100,
        direction="at_most",
        numerical_error=0.005,
        missing_category="numerical",
    ),
    _threshold(
        "beam-fault-aperture-boundary",
        "beam",
        "Radius is too close to the aperture limit",
        True,
        0.095,
        0.100,
        direction="at_most",
        numerical_error=0.010,
    ),
    _threshold(
        "beam-fault-unscoped-window",
        "beam",
        "Radius omits the finite-window scope",
        True,
        0.020,
        0.100,
        direction="at_most",
        numerical_error=0.005,
        missing_category="scope",
    ),
    _choice(
        "decision-control-cavity",
        "decision",
        "Cavity design has a stable best candidate",
        False,
        (("design-a", 0.91), ("design-b", 0.70)),
        error_budget=0.05,
    ),
    _choice(
        "decision-control-mie",
        "decision",
        "Particle design has a stable best candidate",
        False,
        (("design-a", 0.74), ("design-b", 0.63)),
        error_budget=0.03,
    ),
    _threshold(
        "decision-control-budget",
        "decision",
        "Thermal budget remains below the declared limit",
        False,
        8.0,
        10.0,
        direction="at_most",
        parameter_error=0.5,
    ),
    _choice(
        "decision-fault-missing-semantic",
        "decision",
        "Field and power representations are not linked",
        True,
        (("design-a", 0.91), ("design-b", 0.70)),
        error_budget=0.05,
        missing_category="semantic",
    ),
    _choice(
        "decision-fault-near-tie",
        "decision",
        "Best design changes under the declared error budget",
        True,
        (("design-a", 0.91), ("design-b", 0.88)),
        error_budget=0.04,
    ),
    _threshold(
        "decision-fault-unproven-provenance",
        "decision",
        "Thermal budget has no source provenance",
        True,
        8.0,
        10.0,
        direction="at_most",
        parameter_error=0.5,
        missing_category="provenance",
    ),
)


def _manifest_hash() -> str:
    payload = [asdict(case) for case in PROTOCOL_MANIFEST]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


MANIFEST_HASH = _manifest_hash()


def _claim(case: ProtocolCaseDefinition) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=case.case_id,
        representation=f"{case.family} decision representation",
        observable="declared decision observable",
        tolerance="positive decision-stability margin",
        diagnostic="compositional evidence protocol",
        evidence_source="independent protocol benchmark reference",
        scope=f"{case.family} benchmark scope",
        requirements=EvidenceRequirements(independent_reference=True),
    )


def _graph(case: ProtocolCaseDefinition) -> EvidenceGraph:
    nodes = [
        EvidenceNode(
            f"{case.case_id}-{category}",
            frozenset({category}) if category != case.missing_category else frozenset(),
            f"{case.family} protocol {category}",
            f"{case.family} benchmark scope",
        )
        for category in _CATEGORIES
    ]
    nodes.append(
        EvidenceNode(
            f"{case.case_id}-result",
            frozenset(),
            f"{case.family} protocol result",
            f"{case.family} benchmark scope",
            tuple(node.node_id for node in nodes),
        )
    )
    return EvidenceGraph(nodes)


def _stability(case: ProtocolCaseDefinition) -> tuple[bool, float]:
    if case.decision_kind == "threshold":
        threshold_result = threshold_stability(
            case.value,
            case.boundary,
            numerical_error=case.numerical_error,
            reference_error=case.reference_error,
            parameter_error=case.parameter_error,
            direction=case.direction,
        )
        return threshold_result.stable, threshold_result.margin
    if case.decision_kind == "choice":
        choice_result = choice_stability(
            dict(case.choice_values),
            error_budget=case.choice_error_budget,
            maximize=case.maximize,
        )
        return choice_result.stable, choice_result.margin
    raise ValueError(f"unknown decision kind: {case.decision_kind}")


def run_protocol_benchmark(
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> list[ProtocolCaseResult]:
    """Evaluate all frozen protocol cases and optionally write CSV artifacts."""
    results: list[ProtocolCaseResult] = []
    for case in PROTOCOL_MANIFEST:
        started = time.perf_counter()
        stability, margin = _stability(case)
        # The graph gate and the stability gate are evaluated separately above
        # so the CSV can identify both failure mechanisms without hiding one.
        graph_decision = _graph(case).evaluate(f"{case.case_id}-result", _claim(case))
        decision_ready = graph_decision.decision_ready and stability
        blocking = list(graph_decision.blocking_reasons)
        if not stability:
            blocking.append("stability")
        blocking_reasons = tuple(dict.fromkeys(blocking))
        detected = case.is_fault and not decision_ready
        expected = decision_ready is (not case.is_fault)
        results.append(
            ProtocolCaseResult(
                case_id=case.case_id,
                family=case.family,
                is_fault=case.is_fault,
                decision_ready=decision_ready,
                detected=detected,
                stable=stability,
                margin=margin,
                blocking_reasons=blocking_reasons,
                manifest_hash=MANIFEST_HASH,
                elapsed_seconds=time.perf_counter() - started,
                expected_outcome_met=expected,
            )
        )

    if output_path is not None or summary_path is not None:
        write_protocol_results(results, output_path or DEFAULT_CASE_OUTPUT_PATH)
        write_protocol_summary(
            protocol_summary(results), summary_path or DEFAULT_SUMMARY_OUTPUT_PATH
        )
    return results


def protocol_summary(rows: Iterable[ProtocolCaseResult]) -> ProtocolSummary:
    """Aggregate detection, acceptance, and incorrect-promotion rates."""
    results = list(rows)
    if not results:
        raise ValueError("protocol summary requires at least one result")
    faults = [row for row in results if row.is_fault]
    controls = [row for row in results if not row.is_fault]
    detected_faults = sum(row.detected for row in faults)
    accepted_controls = sum(row.decision_ready for row in controls)
    faulty_decision_ready = sum(row.decision_ready for row in faults)
    return ProtocolSummary(
        total_cases=len(results),
        faults=len(faults),
        controls=len(controls),
        detected_faults=detected_faults,
        accepted_controls=accepted_controls,
        faulty_decision_ready=faulty_decision_ready,
        fault_detection_rate=detected_faults / len(faults) if faults else 0.0,
        control_acceptance_rate=accepted_controls / len(controls) if controls else 0.0,
        incorrect_promotion_rate=faulty_decision_ready / len(faults) if faults else 0.0,
        manifest_hash=results[0].manifest_hash,
    )


def write_protocol_results(
    rows: Iterable[ProtocolCaseResult], output_path: str | Path = DEFAULT_CASE_OUTPUT_PATH
) -> Path:
    """Write row-level protocol outcomes to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "case_id",
        "family",
        "is_fault",
        "decision_ready",
        "detected",
        "stable",
        "margin",
        "blocking_reasons",
        "manifest_hash",
        "elapsed_seconds",
        "expected_outcome_met",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["blocking_reasons"] = ";".join(row.blocking_reasons)
            writer.writerow(payload)
    return path


def write_protocol_summary(
    summary: ProtocolSummary, output_path: str | Path = DEFAULT_SUMMARY_OUTPUT_PATH
) -> Path:
    """Write aggregate protocol metrics to a one-row CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(asdict(summary)))
        writer.writeheader()
        writer.writerow(asdict(summary))
    return path


def main() -> int:
    """Run the protocol suite and return a process-style status code."""
    rows = run_protocol_benchmark(
        output_path=DEFAULT_CASE_OUTPUT_PATH,
        summary_path=DEFAULT_SUMMARY_OUTPUT_PATH,
    )
    summary = protocol_summary(rows)
    print(
        "compositional protocol: "
        f"{summary.detected_faults}/{summary.faults} faults detected, "
        f"{summary.accepted_controls}/{summary.controls} controls accepted, "
        f"{summary.faulty_decision_ready} faulty decisions promoted"
    )
    print(f"manifest hash: {summary.manifest_hash}")
    print(f"wrote {DEFAULT_CASE_OUTPUT_PATH}")
    print(f"wrote {DEFAULT_SUMMARY_OUTPUT_PATH}")
    return 0 if all(row.expected_outcome_met for row in rows) else 1


__all__ = [
    "DEFAULT_CASE_OUTPUT_PATH",
    "DEFAULT_SUMMARY_OUTPUT_PATH",
    "MANIFEST_HASH",
    "PROTOCOL_MANIFEST",
    "ProtocolCaseDefinition",
    "ProtocolCaseResult",
    "ProtocolSummary",
    "main",
    "protocol_summary",
    "run_protocol_benchmark",
    "write_protocol_results",
    "write_protocol_summary",
]


if __name__ == "__main__":
    raise SystemExit(main())
