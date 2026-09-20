"""Classify the declared fault corpus by detector and detection boundary.

The coverage table is intentionally narrower than a bug taxonomy for all
optical software. It records what the current corpus exercises and identifies
where matched reference evidence is required.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .fault_injection import FaultInjectionResult, run_fault_injection_benchmark

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "optcon-artifacts"
    / "benchmarks"
    / "contract_coverage.csv"
)
DEFAULT_CLAIMS_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "optcon-artifacts"
    / "benchmarks"
    / "verification_claims.csv"
)


@dataclass(frozen=True)
class ContractCoverageRow:
    """One declared case and the boundary at which it can be diagnosed."""

    case_id: str
    family: str
    is_fault: bool
    detector_layer: str
    primary_boundary: str
    detectable_by_local_contract: bool
    requires_external_reference: bool


@dataclass(frozen=True)
class VerificationClaim:
    """A traceable claim-to-evidence record for one declared benchmark case."""

    case_id: str
    claim_type: str
    representation: str
    observable: str
    tolerance: str
    detector_layer: str
    evidence_source: str
    scope: str
    outcome: str


_BOUNDARY_BY_CASE: dict[str, tuple[str, bool, bool]] = {
    "control-compatible-lengths": ("representation", True, False),
    "control-power-to-field": ("representation", True, False),
    "control-lossless-qwp": ("operator", True, False),
    "control-reciprocal-scatterer": ("operator", True, False),
    "control-complex-adjoint": ("adjoint", True, False),
    "control-real-parameter-adjoint": ("adjoint", True, False),
    "control-normalized-fft": ("operator", True, False),
    "fault-angle-added-to-reflectance": ("representation", True, False),
    "fault-field-power-addition": ("representation", True, False),
    "fault-repeated-square-root": ("representation", True, False),
    "fault-lossy-declared-unitary": ("operator", True, False),
    "fault-active-declared-passive": ("operator", True, False),
    "fault-asymmetric-declared-reciprocal": ("operator", True, False),
    "fault-missing-complex-conjugation": ("adjoint", True, False),
    "fault-complex-cotangent-for-real-parameter": ("adjoint", True, False),
    "fault-omitted-quadrature-metric": ("adjoint", True, False),
    "fault-unnormalized-fft-declared-unitary": ("operator", True, False),
    "fault-mie-medium-convention": ("engine-convention", False, True),
}


_CLAIM_METADATA_BY_CASE: dict[str, tuple[str, str, str, str, str]] = {
    "control-compatible-lengths": (
        "length-valued quantities",
        "dimensional equality after addition",
        "exact dimension match",
        "local quantity contract",
        "compatible length conversion and addition",
    ),
    "control-power-to-field": (
        "power transmission and field amplitude",
        "amplitude order after square root",
        "even non-negative order required",
        "local amplitude-provenance contract",
        "one declared power-to-field conversion",
    ),
    "control-lossless-qwp": (
        "complex Jones operator in an orthonormal basis",
        "unitarity residual",
        "rtol=1e-9, atol=1e-12",
        "local operator predicate",
        "declared lossless operator in the supplied basis",
    ),
    "control-reciprocal-scatterer": (
        "two-port complex operator in a matched reciprocal basis",
        "transpose symmetry residual",
        "rtol=1e-9, atol=1e-12",
        "local operator predicate",
        "declared symmetric port normalization",
    ),
    "control-complex-adjoint": (
        "complex-linear map and Hermitian reverse map",
        "seeded dot-product residual",
        "dot-test default relative tolerance",
        "local discrete-adjoint test",
        "one finite-dimensional complex operator",
    ),
    "control-real-parameter-adjoint": (
        "real parameter map into a complex field",
        "real dual pairing and cotangent domain",
        "dot-test default relative tolerance",
        "local discrete-adjoint and domain test",
        "one real-parameter discretization",
    ),
    "control-normalized-fft": (
        "orthonormally scaled discrete Fourier operator",
        "unitarity residual",
        "rtol=1e-9, atol=1e-12",
        "local operator predicate",
        "declared orthonormal discrete transform",
    ),
    "fault-angle-added-to-reflectance": (
        "angle-valued tilt and dimensionless reflectance",
        "dimensional equality at addition",
        "exact dimension match",
        "local quantity contract",
        "one incompatible scalar addition",
    ),
    "fault-field-power-addition": (
        "field amplitude and power transmission",
        "amplitude-order equality at addition",
        "matching tracked order",
        "local amplitude-provenance contract",
        "one field/power substitution",
    ),
    "fault-repeated-square-root": (
        "field amplitude subjected to a second square root",
        "amplitude order at square root",
        "even non-negative order required",
        "local amplitude-provenance contract",
        "one repeated field conversion",
    ),
    "fault-lossy-declared-unitary": (
        "lossy complex Jones operator",
        "unitarity residual",
        "rtol=1e-9, atol=1e-12",
        "local operator predicate",
        "one declared lossless operator",
    ),
    "fault-active-declared-passive": (
        "amplifying scalar operator",
        "largest singular value above unity",
        "rtol=1e-9, atol=1e-12",
        "local operator predicate",
        "one declared passive operator",
    ),
    "fault-asymmetric-declared-reciprocal": (
        "asymmetric two-port operator",
        "transpose symmetry residual",
        "rtol=1e-9, atol=1e-12",
        "local operator predicate",
        "one matched-basis reciprocity assertion",
    ),
    "fault-missing-complex-conjugation": (
        "complex-linear map with transpose-only reverse map",
        "Hermitian dot-product residual",
        "dot-test default relative tolerance",
        "local discrete-adjoint test",
        "one finite-dimensional complex operator",
    ),
    "fault-complex-cotangent-for-real-parameter": (
        "real parameter map with complex cotangent",
        "cotangent domain residual",
        "real parameter space required",
        "local discrete-adjoint and domain test",
        "one real-parameter discretization",
    ),
    "fault-omitted-quadrature-metric": (
        "Gauss--Legendre weighted representation and uniform-metric reverse map",
        "weighted dot-product residual",
        "dot-test default relative tolerance",
        "local discrete-adjoint test",
        "one nonuniform quadrature discretization",
    ),
    "fault-unnormalized-fft-declared-unitary": (
        "unnormalized discrete Fourier operator",
        "unitarity residual",
        "rtol=1e-9, atol=1e-12",
        "local operator predicate",
        "one declared orthonormal transform",
    ),
    "fault-mie-medium-convention": (
        "Mie query with vacuum versus air surrounding medium",
        "shared extinction-efficiency discrepancy",
        "relative tolerance rtol=1e-6",
        "author-constructed series plus external-engine matched query",
        "one sphere query over the declared five-diameter sweep",
    ),
}


def _detector_layer(result: FaultInjectionResult) -> str:
    if not result.is_fault:
        return "accepted-control"
    if result.full_contract_detected:
        return "full-contract"
    if result.dimension_only_detected:
        return "dimensions-only"
    if result.numeric_smoke_detected:
        return "numeric-smoke"
    return "undetected"


def build_contract_coverage(
    results: Iterable[FaultInjectionResult] | None = None,
) -> list[ContractCoverageRow]:
    """Build deterministic coverage rows for the declared fault corpus."""
    selected = list(
        run_fault_injection_benchmark(repeats=1) if results is None else results
    )
    rows: list[ContractCoverageRow] = []
    for result in selected:
        try:
            boundary, locally_detectable, requires_reference = _BOUNDARY_BY_CASE[
                result.case_id
            ]
        except KeyError as error:
            raise ValueError(f"no coverage boundary for {result.case_id!r}") from error
        rows.append(
            ContractCoverageRow(
                case_id=result.case_id,
                family=result.family,
                is_fault=result.is_fault,
                detector_layer=_detector_layer(result),
                primary_boundary=boundary,
                detectable_by_local_contract=locally_detectable,
                requires_external_reference=requires_reference,
            )
        )
    return rows


def summarize_contract_coverage(
    rows: Sequence[ContractCoverageRow],
) -> dict[str, Any]:
    """Summarize corpus counts without interpreting them as population rates."""
    faults = [row for row in rows if row.is_fault]
    controls = [row for row in rows if not row.is_fault]
    return {
        "cases_total": len(rows),
        "faults_total": len(faults),
        "controls_total": len(controls),
        "local_faults": sum(
            row.detectable_by_local_contract and not row.requires_external_reference
            for row in faults
        ),
        "reference_dependent_faults": sum(
            row.requires_external_reference for row in faults
        ),
        "boundaries": {row.primary_boundary for row in faults},
    }


def build_verification_claims(
    results: Iterable[FaultInjectionResult] | None = None,
) -> list[VerificationClaim]:
    """Build an auditable claim-to-evidence ledger for the declared corpus.

    The ledger is deliberately case-scoped.  It records what was asserted and
    what evidence was collected without turning the declared corpus into a
    population-level accuracy estimate.
    """
    selected = list(
        run_fault_injection_benchmark(repeats=1) if results is None else results
    )
    claims: list[VerificationClaim] = []
    for result in selected:
        try:
            representation, observable, tolerance, evidence_source, scope = (
                _CLAIM_METADATA_BY_CASE[result.case_id]
            )
        except KeyError as error:
            raise ValueError(f"no claim metadata for {result.case_id!r}") from error

        coverage = build_contract_coverage([result])[0]
        detector_layer = (
            "external-reference"
            if coverage.requires_external_reference
            else coverage.detector_layer
        )
        claims.append(
            VerificationClaim(
                case_id=result.case_id,
                claim_type="fault-detection" if result.is_fault else "valid-control",
                representation=representation,
                observable=observable,
                tolerance=tolerance,
                detector_layer=detector_layer,
                evidence_source=evidence_source,
                scope=scope,
                outcome=(
                    "detected"
                    if result.is_fault and result.full_contract_detected
                    else "accepted"
                ),
            )
        )
    return claims


def write_verification_claims(
    claims: Sequence[VerificationClaim],
    output_path: str | Path,
) -> Path:
    """Write the claim-to-evidence ledger as a machine-readable CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(VerificationClaim.__annotations__),
        )
        writer.writeheader()
        writer.writerows(asdict(claim) for claim in claims)
    return path


def write_contract_coverage(
    rows: Sequence[ContractCoverageRow],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write the coverage table as an auditable CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(ContractCoverageRow.__annotations__),
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return path


def main() -> int:
    results = run_fault_injection_benchmark(repeats=1)
    rows = build_contract_coverage(results)
    write_contract_coverage(rows)
    claims = build_verification_claims(results)
    write_verification_claims(claims, DEFAULT_CLAIMS_OUTPUT_PATH)
    summary = summarize_contract_coverage(rows)
    print(
        "contract coverage: "
        f"{summary['faults_total']} faults, "
        f"{summary['controls_total']} controls, "
        f"{summary['reference_dependent_faults']} reference-dependent faults"
    )
    print(f"wrote {DEFAULT_OUTPUT_PATH}")
    print(f"wrote {DEFAULT_CLAIMS_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
