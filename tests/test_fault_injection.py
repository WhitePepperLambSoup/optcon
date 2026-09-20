import csv

import pytest

from optcon.benchmarks.fault_injection import (
    DEFAULT_FAULT_INJECTION_PATH,
    run_fault_injection_benchmark,
    summarize_fault_injection,
    write_fault_injection_results,
)


def _outcomes(results):
    return [
        (
            result.case_id,
            result.is_fault,
            result.numeric_smoke_detected,
            result.dimension_only_detected,
            result.full_contract_detected,
            result.diagnostic_type,
        )
        for result in results
    ]


def test_fault_injection_benchmark_has_controls_and_diverse_defect_families():
    results = run_fault_injection_benchmark(repeats=1)
    faults = [result for result in results if result.is_fault]
    controls = [result for result in results if not result.is_fault]

    assert len(faults) >= 10
    assert len(controls) >= 4
    assert len({result.case_id for result in results}) == len(results)
    assert {
        "dimensions",
        "amplitude_provenance",
        "unitarity",
        "passivity",
        "reciprocity",
        "adjoint_conjugation",
        "adjoint_domain",
        "quadrature_metric",
        "fft_normalization",
        "engine_convention",
    } <= {result.family for result in faults}


def test_full_contract_layer_detects_all_injected_faults_without_false_positives():
    results = run_fault_injection_benchmark(repeats=1)

    assert all(result.full_contract_detected for result in results if result.is_fault)
    assert not any(
        result.full_contract_detected for result in results if not result.is_fault
    )


def test_adjoint_fault_effect_sizes_are_measured_residuals():
    results = {
        result.case_id: result for result in run_fault_injection_benchmark(repeats=1)
    }

    missing_conjugate = results["fault-missing-complex-conjugation"]
    complex_cotangent = results["fault-complex-cotangent-for-real-parameter"]
    quadrature_metric = results["fault-omitted-quadrature-metric"]

    assert missing_conjugate.effect_value == pytest.approx(1.0456640102394434)
    assert complex_cotangent.effect_value == pytest.approx(0.6327466718234475)
    assert quadrature_metric.effect_value == pytest.approx(0.011091256092915787)


def test_baselines_miss_semantically_silent_defects():
    summary = summarize_fault_injection(run_fault_injection_benchmark(repeats=1))

    assert summary["faults_total"] >= 10
    assert summary["numeric_smoke_detected"] == 0
    assert 0 < summary["dimension_only_detected"] < summary["faults_total"]
    assert summary["full_contract_detected"] == summary["faults_total"]
    assert summary["controls_total"] >= 4
    assert summary["full_contract_false_positives"] == 0


def test_fault_injection_outcomes_are_deterministic():
    first = run_fault_injection_benchmark(repeats=1)
    second = run_fault_injection_benchmark(repeats=1)

    assert _outcomes(first) == _outcomes(second)


def test_fault_injection_results_can_be_written_as_auditable_csv(tmp_path):
    results = run_fault_injection_benchmark(repeats=1)
    output = tmp_path / "fault-injection.csv"

    written = write_fault_injection_results(results, output)

    assert written == output
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(results)
    assert rows[0]["case_id"]
    assert {row["is_fault"] for row in rows} == {"True", "False"}
    assert all(float(row["full_contract_runtime_us"]) >= 0.0 for row in rows)


def test_default_fault_injection_output_is_not_a_documentation_path():
    assert "optcon-artifacts" in DEFAULT_FAULT_INJECTION_PATH.parts
    assert "docs" not in DEFAULT_FAULT_INJECTION_PATH.parts
