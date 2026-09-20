from optcon.benchmarks.contract_coverage import (
    DEFAULT_CLAIMS_OUTPUT_PATH,
    build_contract_coverage,
    build_verification_claims,
    summarize_contract_coverage,
    write_verification_claims,
)
from optcon.benchmarks.fault_injection import run_fault_injection_benchmark


def test_contract_coverage_has_a_boundary_for_every_declared_case():
    rows = build_contract_coverage(run_fault_injection_benchmark(repeats=1))

    assert rows
    assert {row.case_id for row in rows} == {
        result.case_id for result in run_fault_injection_benchmark(repeats=1)
    }
    assert all(row.primary_boundary for row in rows)
    assert all(row.detector_layer for row in rows)


def test_contract_coverage_distinguishes_local_and_reference_dependent_faults():
    rows = build_contract_coverage(run_fault_injection_benchmark(repeats=1))
    summary = summarize_contract_coverage(rows)

    assert summary["faults_total"] == 11
    assert summary["controls_total"] == 7
    assert summary["local_faults"] == 10
    assert summary["reference_dependent_faults"] == 1
    assert summary["boundaries"] == {
        "adjoint",
        "engine-convention",
        "operator",
        "representation",
    }


def test_verification_claims_make_each_case_traceable(tmp_path):
    results = run_fault_injection_benchmark(repeats=1)

    claims = build_verification_claims(results)

    assert len(claims) == len(results)
    assert {claim.case_id for claim in claims} == {
        result.case_id for result in results
    }
    assert all(claim.representation for claim in claims)
    assert all(claim.observable for claim in claims)
    assert all(claim.tolerance for claim in claims)
    assert all(claim.detector_layer for claim in claims)
    assert all(claim.evidence_source for claim in claims)
    assert all(claim.scope for claim in claims)

    mie_claim = next(
        claim for claim in claims if claim.case_id == "fault-mie-medium-convention"
    )
    assert mie_claim.detector_layer == "external-reference"
    evidence_source = mie_claim.evidence_source.lower()
    assert "matched" in evidence_source
    assert "external" in evidence_source

    output = write_verification_claims(claims, tmp_path / "verification_claims.csv")
    assert output.is_file()


def test_default_claim_ledger_is_an_artifact_path():
    assert "optcon-artifacts" in DEFAULT_CLAIMS_OUTPUT_PATH.parts
    assert "private-paper" not in DEFAULT_CLAIMS_OUTPUT_PATH.parts
