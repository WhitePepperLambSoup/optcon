"""Experiment 06 - Mie medium conventions and decision evidence.

The candidate engine is evaluated against the author-constructed Mie series in
the evaluated source tree for a
declared water-medium query.  A second path uses the vacuum default while the
design question still requires water.  Both paths return finite efficiencies,
but only the harmonised path is promoted to a threshold decision.

Run:
    python -m optcon.examples.experiment_06_mie_decision_impact
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from optcon import (
    EvidenceClaim,
    EvidenceRecord,
    EvidenceRequirements,
    evaluate_claim,
    q,
)
from optcon.engines import available_engines, mie_efficiencies
from optcon.engines.reference_mie import mie_efficiencies_reference

MATERIAL_INDEX = 1.5 + 0.01j
WAVELENGTH_NM = 550.0
WATER_INDEX = 1.33
VACUUM_INDEX = 1.0
TARGET_QEXT = 0.5
DIAMETERS_NM = np.linspace(20.0, 1000.0, 981)
REFERENCE_RTOL = 1.0e-6


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    scale = np.maximum(np.abs(value), np.abs(reference))
    residual = np.divide(
        np.abs(value - reference),
        scale,
        out=np.zeros_like(value, dtype=float),
        where=scale != 0.0,
    )
    return float(np.max(residual))


def _first_upward_crossing(axis: np.ndarray, values: np.ndarray, level: float) -> float:
    """Interpolate the first crossing of an increasing threshold."""
    for index in range(1, len(axis)):
        if values[index - 1] < level <= values[index]:
            left_x = float(axis[index - 1])
            right_x = float(axis[index])
            left_y = float(values[index - 1])
            right_y = float(values[index])
            if right_y == left_y:
                return (left_x + right_x) / 2.0
            fraction = (level - left_y) / (right_y - left_y)
            return left_x + fraction * (right_x - left_x)
    raise ValueError(f"no upward crossing found for level {level}")


def _candidate_engine() -> str:
    for name in ("miepython", "PyMieScatt"):
        if name in available_engines("mie"):
            return name
    raise RuntimeError(
        "Experiment 06 requires an external Mie engine; install optcon[engines]"
    )


def _candidate_sweep(engine: str, medium_index: float) -> np.ndarray:
    return np.asarray(
        [
            mie_efficiencies(
                m=MATERIAL_INDEX,
                diameter=q(float(diameter), "nm"),
                wavelength=q(WAVELENGTH_NM, "nm"),
                medium_index=medium_index,
                engine=engine,
            )["Qext"].value
            for diameter in DIAMETERS_NM
        ],
        dtype=float,
    )


def _reference_sweep(medium_index: float) -> np.ndarray:
    return np.asarray(
        [
            mie_efficiencies_reference(
                MATERIAL_INDEX,
                float(diameter),
                WAVELENGTH_NM,
                n_env=medium_index,
            )["Qext"]
            for diameter in DIAMETERS_NM
        ],
        dtype=float,
    )


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_decision_impact(output_path: str | Path) -> dict[str, Any]:
    """Run the Mie convention comparison and export row-level evidence."""
    engine = _candidate_engine()
    water_reference = _reference_sweep(WATER_INDEX)
    water_candidate = _candidate_sweep(engine, WATER_INDEX)
    vacuum_candidate = _candidate_sweep(engine, VACUUM_INDEX)

    water_error = _relative_error(water_candidate, water_reference)
    vacuum_error = _relative_error(vacuum_candidate, water_reference)
    water_threshold = _first_upward_crossing(DIAMETERS_NM, water_candidate, TARGET_QEXT)
    vacuum_threshold = _first_upward_crossing(DIAMETERS_NM, vacuum_candidate, TARGET_QEXT)
    water_finite = bool(np.isfinite(water_candidate).all())
    vacuum_finite = bool(np.isfinite(vacuum_candidate).all())

    requirements = EvidenceRequirements(independent_reference=True)
    evidence_claim = EvidenceClaim(
        claim_id="mie-water-threshold",
        representation="relative refractive index and in-medium wavelength",
        observable="Qext threshold diameter",
        tolerance="maximum relative reference error <= 1e-6",
        diagnostic="matched Mie series and external-engine comparison",
        evidence_source=(
            "author-constructed Mie series and external candidate engine"
        ),
        scope="homogeneous sphere in a declared water medium",
        requirements=requirements,
    )
    water_gate = evaluate_claim(
        evidence_claim,
        EvidenceRecord(
            semantic=True,
            numerical=water_finite,
            independent_reference=water_error <= REFERENCE_RTOL,
            provenance=True,
            scope=True,
        ),
    )
    vacuum_gate = evaluate_claim(
        evidence_claim,
        EvidenceRecord(
            semantic=True,
            numerical=vacuum_finite,
            independent_reference=vacuum_error <= REFERENCE_RTOL,
            provenance=True,
            scope=True,
        ),
    )

    rows = [
        {
            "case_id": "declared-water-medium",
            "claim_id": evidence_claim.claim_id,
            "representation": evidence_claim.representation,
            "observable": evidence_claim.observable,
            "tolerance": evidence_claim.tolerance,
            "diagnostic": evidence_claim.diagnostic,
            "evidence_source": evidence_claim.evidence_source,
            "claim_scope": evidence_claim.scope,
            "candidate_engine": engine,
            "declared_medium_index": f"{WATER_INDEX:.12g}",
            "reference_medium_index": f"{WATER_INDEX:.12g}",
            "threshold_qext": f"{TARGET_QEXT:.12g}",
            "threshold_diameter_nm": f"{water_threshold:.12g}",
            "max_relative_reference_error": f"{water_error:.12g}",
            "finite_result": water_finite,
            "semantic_evidence": True,
            "numerical_evidence": water_finite,
            "independent_reference_evidence": water_error <= REFERENCE_RTOL,
            "provenance_evidence": True,
            "scope_evidence": True,
            "status": water_gate.status,
            "required_evidence": ";".join(water_gate.required_categories),
            "available_evidence": ";".join(water_gate.available_categories),
            "decision_ready": water_gate.decision_ready,
            "failed_requirements": ";".join(water_gate.failed_requirements),
        },
        {
            "case_id": "vacuum-default-mismatch",
            "claim_id": evidence_claim.claim_id,
            "representation": evidence_claim.representation,
            "observable": evidence_claim.observable,
            "tolerance": evidence_claim.tolerance,
            "diagnostic": evidence_claim.diagnostic,
            "evidence_source": evidence_claim.evidence_source,
            "claim_scope": evidence_claim.scope,
            "candidate_engine": engine,
            "declared_medium_index": f"{VACUUM_INDEX:.12g}",
            "reference_medium_index": f"{WATER_INDEX:.12g}",
            "threshold_qext": f"{TARGET_QEXT:.12g}",
            "threshold_diameter_nm": f"{vacuum_threshold:.12g}",
            "max_relative_reference_error": f"{vacuum_error:.12g}",
            "finite_result": vacuum_finite,
            "semantic_evidence": True,
            "numerical_evidence": vacuum_finite,
            "independent_reference_evidence": vacuum_error <= REFERENCE_RTOL,
            "provenance_evidence": True,
            "scope_evidence": True,
            "status": vacuum_gate.status,
            "required_evidence": ";".join(vacuum_gate.required_categories),
            "available_evidence": ";".join(vacuum_gate.available_categories),
            "decision_ready": vacuum_gate.decision_ready,
            "failed_requirements": ";".join(vacuum_gate.failed_requirements),
        },
    ]
    _write_csv(
        Path(output_path),
        rows,
        (
            "case_id",
            "claim_id",
            "representation",
            "observable",
            "tolerance",
            "diagnostic",
            "evidence_source",
            "claim_scope",
            "candidate_engine",
            "declared_medium_index",
            "reference_medium_index",
            "threshold_qext",
            "threshold_diameter_nm",
            "max_relative_reference_error",
            "finite_result",
            "semantic_evidence",
            "numerical_evidence",
            "independent_reference_evidence",
            "provenance_evidence",
            "scope_evidence",
            "status",
            "required_evidence",
            "available_evidence",
            "decision_ready",
            "failed_requirements",
        ),
    )
    return {
        "candidate_engine": engine,
        "correct_decision_ready": water_gate.decision_ready,
        "fault_decision_ready": vacuum_gate.decision_ready,
        "fault_finite": vacuum_finite,
        "correct_threshold_diameter_nm": water_threshold,
        "fault_threshold_diameter_nm": vacuum_threshold,
        "correct_max_relative_reference_error": water_error,
        "fault_max_relative_reference_error": vacuum_error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "optcon-artifacts" / "mie_decision_impact.csv",
        help="path for the row-level CSV output",
    )
    args = parser.parse_args(argv)
    result = run_decision_impact(args.output)
    print(f"candidate engine: {result['candidate_engine']}")
    print(
        "water threshold: "
        f"{result['correct_threshold_diameter_nm']:.2f} nm; "
        f"decision-ready={result['correct_decision_ready']}"
    )
    print(
        "vacuum threshold: "
        f"{result['fault_threshold_diameter_nm']:.2f} nm; "
        f"decision-ready={result['fault_decision_ready']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
