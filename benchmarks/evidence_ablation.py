"""Ablate decision evidence requirements for the manuscript evaluation.

The experiment asks whether the category that blocks each declared faulty
decision path is specific to that path.  It is a protocol check, not a claim
about the prevalence of missing evidence in external software.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from optcon.evidence import EvidenceRecord, EvidenceRequirements, evaluate_evidence

EVIDENCE_CATEGORIES = (
    "semantic",
    "numerical",
    "independent_reference",
    "provenance",
    "scope",
)

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "optcon-artifacts"
    / "benchmarks"
    / "evidence_requirement_ablation.csv"
)


@dataclass(frozen=True)
class EvidenceRequirementAblation:
    """One one-category removal from a declared decision requirement."""

    source_case: str
    case_id: str
    removed_requirement: str
    full_decision_ready: bool
    ablated_decision_ready: bool
    released_by_removal: bool


_FAULT_CASES = (
    (
        "Fabry-Perot",
        "fault-power-used-as-field",
        EvidenceRecord(
            semantic=False,
            numerical=True,
            independent_reference=False,
            provenance=True,
            scope=True,
        ),
        EvidenceRequirements(),
    ),
    (
        "Mie medium",
        "vacuum-default-mismatch",
        EvidenceRecord(
            semantic=True,
            numerical=True,
            independent_reference=False,
            provenance=True,
            scope=True,
        ),
        EvidenceRequirements(independent_reference=True),
    ),
    (
        "Thin-film stack",
        "omitted-terminal-high",
        EvidenceRecord(
            semantic=True,
            numerical=True,
            independent_reference=False,
            provenance=True,
            scope=True,
        ),
        EvidenceRequirements(independent_reference=True),
    ),
)


def run_evidence_requirement_ablation(
    output_path: str | Path | None = None,
) -> list[EvidenceRequirementAblation]:
    """Evaluate every one-category requirement removal for each fault path."""
    rows: list[EvidenceRequirementAblation] = []
    for source_case, case_id, record, requirements in _FAULT_CASES:
        full = evaluate_evidence(record, requirements).decision_ready
        for category in EVIDENCE_CATEGORIES:
            ablated_requirements = replace(requirements, **{category: False})
            ablated = evaluate_evidence(record, ablated_requirements).decision_ready
            rows.append(
                EvidenceRequirementAblation(
                    source_case=source_case,
                    case_id=case_id,
                    removed_requirement=category,
                    full_decision_ready=full,
                    ablated_decision_ready=ablated,
                    released_by_removal=bool(ablated and not full),
                )
            )

    if output_path is not None:
        write_evidence_requirement_ablation(rows, output_path)
    return rows


def write_evidence_requirement_ablation(
    rows: Iterable[EvidenceRequirementAblation],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write row-level requirement-ablation results to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(EvidenceRequirementAblation.__annotations__)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return path


def main() -> int:
    rows = run_evidence_requirement_ablation(output_path=DEFAULT_OUTPUT_PATH)
    released = [row for row in rows if row.released_by_removal]
    print(
        "evidence requirement ablation: "
        f"{len(released)}/{len(rows)} removals release a faulty decision"
    )
    print(f"wrote {DEFAULT_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
