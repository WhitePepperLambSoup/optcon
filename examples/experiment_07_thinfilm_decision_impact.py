"""Experiment 07 - Thin-film stack convention and decision evidence.

The declared coating is a normal-incidence quarter-wave ``(HL)^N H`` stack.
The valid path uses the closed-form Bragg expression and an external TMM query
with the same layer sequence.  The fault path omits the terminal high
index layer.  Both paths are finite, but the omission changes the minimum pair
count needed to reach the declared reflectance target and is blocked by the
matched-reference requirement.

Run:
    python -m optcon.examples.experiment_07_thinfilm_decision_impact
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from optcon import EvidenceClaim, EvidenceRecord, EvidenceRequirements, evaluate_claim, q
from optcon.engines import available_engines
from optcon.engines.thinfilm import stack_response
from optcon.thinfilm import bragg_reflectance

N_INCIDENT = 1.0
N_HIGH = 2.3
N_LOW = 1.45
N_SUBSTRATE = 1.5
DESIGN_WAVELENGTH_NM = 1064.0
TARGET_REFLECTANCE = 0.995
REFERENCE_ATOL = 1.0e-12
PAIR_COUNTS = np.arange(1, 13, dtype=int)


def _quarter_wave_reflectance(
    n_incident: float, layers: list[float], n_substrate: float
) -> float:
    """Return the normal-incidence characteristic-matrix reflectance."""
    matrix = np.eye(2, dtype=complex)
    for refractive_index in layers:
        matrix = matrix @ np.array(
            [
                [0.0, 1.0j / refractive_index],
                [1.0j * refractive_index, 0.0],
            ],
            dtype=complex,
        )
    a, b = matrix[0]
    c, d = matrix[1]
    input_admittance = (c + d * n_substrate) / (a + b * n_substrate)
    amplitude_reflectance = (n_incident - input_admittance) / (
        n_incident + input_admittance
    )
    return float(abs(amplitude_reflectance) ** 2)


def _declared_layers(pairs: int) -> list[float]:
    """Return the declared ``(HL)^N H`` internal layer sequence."""
    return [value for _ in range(pairs) for value in (N_HIGH, N_LOW)] + [N_HIGH]


def _fault_layers(pairs: int) -> list[float]:
    """Return the mutated ``(HL)^N`` sequence with the terminal layer omitted."""
    return [value for _ in range(pairs) for value in (N_HIGH, N_LOW)]


def _tmm_reflectance(layers: list[float]) -> float:
    thicknesses = [
        q(DESIGN_WAVELENGTH_NM / (4.0 * refractive_index), "nm")
        for refractive_index in layers
    ]
    response = stack_response(
        n_list=[N_INCIDENT, *layers, N_SUBSTRATE],
        thicknesses=[math.inf, *thicknesses, math.inf],
        wavelength=q(DESIGN_WAVELENGTH_NM, "nm"),
        angle=q(0.0, "deg"),
        polarization="s",
        engine="tmm_core",
    )
    return float(response["R"].value)


def _first_threshold(pair_counts: np.ndarray, values: np.ndarray, target: float) -> int:
    """Return the first integer pair count whose value reaches ``target``."""
    matches = pair_counts[values >= target]
    if not len(matches):
        raise ValueError(f"no pair count reaches reflectance target {target}")
    return int(matches[0])


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
    """Run the thin-film sequence comparison and export its evidence trace."""
    if "tmm_core" not in available_engines("thin_film"):
        raise RuntimeError(
            "Experiment 07 requires tmm_core; install optcon[engines] or the "
            "registered TMM source tree"
        )

    closed_form = np.asarray(
        [
            bragg_reflectance(
                N_INCIDENT, N_HIGH, N_LOW, N_SUBSTRATE, int(pairs)
            )
            for pairs in PAIR_COUNTS
        ],
        dtype=float,
    )
    declared_reference = np.asarray(
        [_tmm_reflectance(_declared_layers(int(pairs))) for pairs in PAIR_COUNTS],
        dtype=float,
    )
    omitted_terminal = np.asarray(
        [
            _quarter_wave_reflectance(
                N_INCIDENT, _fault_layers(int(pairs)), N_SUBSTRATE
            )
            for pairs in PAIR_COUNTS
        ],
        dtype=float,
    )
    declared_error = np.abs(closed_form - declared_reference)
    fault_error = np.abs(omitted_terminal - declared_reference)
    correct_pairs = _first_threshold(PAIR_COUNTS, closed_form, TARGET_REFLECTANCE)
    fault_pairs = _first_threshold(PAIR_COUNTS, omitted_terminal, TARGET_REFLECTANCE)
    correct_finite = bool(np.isfinite(closed_form).all())
    fault_finite = bool(np.isfinite(omitted_terminal).all())

    evidence_claim = EvidenceClaim(
        claim_id="bragg-terminal-layer-threshold",
        representation="normal-incidence quarter-wave stack sequence (HL)^N H",
        observable="minimum high-low pair count for a reflectance threshold",
        tolerance="maximum absolute TMM reflectance discrepancy <= 1e-12",
        diagnostic="independent transfer-matrix adjudication",
        evidence_source="optcon closed form and tmm_core",
        scope="isotropic, coherent, lossless quarter-wave Bragg mirror at normal incidence",
        requirements=EvidenceRequirements(independent_reference=True),
    )
    correct_gate = evaluate_claim(
        evidence_claim,
        EvidenceRecord(
            semantic=True,
            numerical=correct_finite,
            independent_reference=bool(np.max(declared_error) <= REFERENCE_ATOL),
            provenance=True,
            scope=True,
        ),
    )
    fault_gate = evaluate_claim(
        evidence_claim,
        EvidenceRecord(
            semantic=True,
            numerical=fault_finite,
            independent_reference=bool(np.max(fault_error) <= REFERENCE_ATOL),
            provenance=True,
            scope=True,
        ),
    )

    rows = [
        {
            "case_id": "declared-terminal-high",
            "claim_id": evidence_claim.claim_id,
            "representation": evidence_claim.representation,
            "observable": evidence_claim.observable,
            "tolerance": evidence_claim.tolerance,
            "diagnostic": evidence_claim.diagnostic,
            "evidence_source": evidence_claim.evidence_source,
            "claim_scope": evidence_claim.scope,
            "contract_status": "accepted",
            "candidate_stack": "(HL)^N H",
            "reference_stack": "(HL)^N H",
            "threshold_reflectance": f"{TARGET_REFLECTANCE:.12g}",
            "reported_pairs": correct_pairs,
            "unchecked_candidate_pairs": "",
            "max_reference_error": f"{float(np.max(declared_error)):.12g}",
            "finite_result": correct_finite,
            "semantic_evidence": True,
            "numerical_evidence": correct_finite,
            "independent_reference_evidence": bool(
                np.max(declared_error) <= REFERENCE_ATOL
            ),
            "provenance_evidence": True,
            "scope_evidence": True,
            "status": correct_gate.status,
            "required_evidence": ";".join(correct_gate.required_categories),
            "available_evidence": ";".join(correct_gate.available_categories),
            "decision_ready": correct_gate.decision_ready,
            "failed_requirements": ";".join(correct_gate.failed_requirements),
        },
        {
            "case_id": "omitted-terminal-high",
            "claim_id": evidence_claim.claim_id,
            "representation": evidence_claim.representation,
            "observable": evidence_claim.observable,
            "tolerance": evidence_claim.tolerance,
            "diagnostic": evidence_claim.diagnostic,
            "evidence_source": evidence_claim.evidence_source,
            "claim_scope": evidence_claim.scope,
            "contract_status": "rejected",
            "candidate_stack": "(HL)^N",
            "reference_stack": "(HL)^N H",
            "threshold_reflectance": f"{TARGET_REFLECTANCE:.12g}",
            "reported_pairs": "",
            "unchecked_candidate_pairs": fault_pairs,
            "max_reference_error": f"{float(np.max(fault_error)):.12g}",
            "finite_result": fault_finite,
            "semantic_evidence": True,
            "numerical_evidence": fault_finite,
            "independent_reference_evidence": bool(
                np.max(fault_error) <= REFERENCE_ATOL
            ),
            "provenance_evidence": True,
            "scope_evidence": True,
            "status": fault_gate.status,
            "required_evidence": ";".join(fault_gate.required_categories),
            "available_evidence": ";".join(fault_gate.available_categories),
            "decision_ready": fault_gate.decision_ready,
            "failed_requirements": ";".join(fault_gate.failed_requirements),
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
            "contract_status",
            "candidate_stack",
            "reference_stack",
            "threshold_reflectance",
            "reported_pairs",
            "unchecked_candidate_pairs",
            "max_reference_error",
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
        "correct_decision_ready": correct_gate.decision_ready,
        "fault_decision_ready": fault_gate.decision_ready,
        "correct_pairs": correct_pairs,
        "fault_pairs": fault_pairs,
        "pair_distortion": fault_pairs - correct_pairs,
        "correct_max_reference_error": float(np.max(declared_error)),
        "fault_max_reference_error": float(np.max(fault_error)),
        "fault_finite": fault_finite,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "optcon-artifacts" / "thinfilm_decision_impact.csv",
        help="path for the row-level CSV output",
    )
    args = parser.parse_args(argv)
    result = run_decision_impact(args.output)
    print(
        "declared stack: "
        f"{result['correct_pairs']} pairs; decision-ready={result['correct_decision_ready']}"
    )
    print(
        "omitted terminal layer: "
        f"{result['fault_pairs']} pairs; decision-ready={result['fault_decision_ready']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
