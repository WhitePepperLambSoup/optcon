import csv
import sys
from pathlib import Path

import numpy as np

from optcon import q
from optcon.benchmarks import (
    adjoint_stress,
    bench,
    convergence,
    fault_injection,
    heldout_mutations,
    run_all,
    thinfilm_adjudication,
)
from optcon.benchmarks.bench import naive_decompose
from optcon.modes import decompose
from optcon.propagation import gaussian_field


def test_default_benchmark_outputs_are_kept_outside_documentation():
    root = Path(__file__).resolve().parents[1]
    artifact_dir = root / "optcon-artifacts" / "benchmarks"

    assert bench.BENCHMARK_DATA_PATH == artifact_dir / "benchmark_operations.csv"
    assert convergence.DEFAULT_OUTPUT_PATH == artifact_dir / "solver_convergence.csv"
    assert fault_injection.DEFAULT_FAULT_INJECTION_PATH == artifact_dir / "fault_injection.csv"
    assert heldout_mutations.DEFAULT_OUTPUT_PATH == artifact_dir / "heldout_mutations.csv"
    assert adjoint_stress.DEFAULT_OUTPUT_PATH == artifact_dir / "adjoint_stress.csv"
    assert thinfilm_adjudication.DEFAULT_OUTPUT_PATH == artifact_dir / "thinfilm_adjudication.csv"


def test_complete_reproducibility_suite_runs_the_fault_corpus():
    assert (
        "Semantic-contract fault-injection corpus",
        "optcon.benchmarks.fault_injection",
    ) in run_all.STAGES


def test_complete_reproducibility_suite_runs_the_heldout_mutation_campaign():
    assert (
        "Frozen held-out mutation campaign",
        "optcon.benchmarks.heldout_mutations",
    ) in run_all.STAGES


def test_complete_reproducibility_suite_runs_the_adjoint_stress_sweep():
    assert (
        "Multi-parameter adjoint stress sweep",
        "optcon.benchmarks.adjoint_stress",
    ) in run_all.STAGES


def test_complete_reproducibility_suite_runs_thinfilm_adjudication():
    assert (
        "Thin-film closed-form and TMM adjudication",
        "optcon.benchmarks.thinfilm_adjudication",
    ) in run_all.STAGES


def test_reproducibility_suite_can_skip_unavailable_optional_stages(monkeypatch, capsys):
    monkeypatch.setattr(run_all, "available_engines", lambda group=None: [])

    assert run_all.stage_available("optcon.benchmarks.thinfilm_adjudication") is False
    assert run_all.run_section(
        "Thin-film closed-form and TMM adjudication",
        "optcon.benchmarks.thinfilm_adjudication",
        skip_unavailable=True,
    ) is True
    assert "[SKIP]" in capsys.readouterr().out


def test_reproducibility_runner_hides_its_arguments_from_stage_main(monkeypatch):
    seen_argv = []

    class Stage:
        @staticmethod
        def main():
            seen_argv.append(list(sys.argv))
            return 0

    monkeypatch.setattr(run_all.importlib, "import_module", lambda _: Stage)
    monkeypatch.setattr(run_all.sys, "argv", ["runner", "--skip-unavailable"])

    assert run_all.run_section("stage", "fake.stage") is True
    assert seen_argv == [["fake.stage"]]
    assert run_all.sys.argv == ["runner", "--skip-unavailable"]


def test_adjoint_stress_sweep_separates_correct_and_wrong_metrics(tmp_path):
    output = tmp_path / "adjoint_stress.csv"

    records = adjoint_stress.run_adjoint_stress(
        output_path=output,
        parameter_count=3,
        probe_seeds=(0, 1, 2),
    )

    assert len(records) == 9
    assert max(record.correct_rel_error for record in records) < 1e-5
    assert min(record.wrong_rel_error for record in records) > 1e-3
    assert all(record.correct_ok for record in records)
    assert all(record.wrong_detected for record in records)
    assert output.exists()


def test_write_results_persists_measurements_with_provenance(tmp_path):
    output = tmp_path / "benchmark.csv"

    bench.write_results([("example", 0.125, 2.5)], output)

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "operation": "example",
            "best_seconds": "0.125",
            "peak_mib": "2.5",
            "python": bench.platform.python_version(),
            "platform": bench.platform.platform(),
        }
    ]


def test_naive_modal_baseline_matches_separable_decomposition():
    field = gaussian_field(
        waist=q(50.0, "um"),
        wavelength=q(633.0, "nm"),
        samples=64,
        extent=q(1600.0, "um"),
    )
    expected = decompose(field, waist=q(50.0, "um"), max_order=2)
    baseline = naive_decompose(field, waist=q(50.0, "um"), max_order=2)
    assert set(baseline) == set(expected)
    assert np.allclose(
        [baseline[key] for key in sorted(baseline)],
        [expected[key] for key in sorted(expected)],
        rtol=1e-12,
        atol=1e-12,
    )
