import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from optcon import q
from optcon.benchmarks import bench
from optcon.benchmarks.bench import naive_decompose
from optcon.modes import decompose
from optcon.propagation import gaussian_field


def _load_figure_generator():
    path = Path(__file__).resolve().parents[1] / "docs" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("optcon_generate_figures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_figure_generator_persists_solver_convergence(monkeypatch, tmp_path):
    module = _load_figure_generator()
    output_path = tmp_path / "solver_convergence.csv"
    calls = []

    def fake_run_solver_convergence(*, output_path):
        calls.append(Path(output_path))
        Path(output_path).write_text("solver\nfox_li\n", encoding="utf-8")
        return [{"solver": "fox_li"}]

    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "run_solver_convergence", fake_run_solver_convergence)

    module.generate_solver_convergence()

    assert calls == [output_path]
    assert output_path.read_text(encoding="utf-8") == "solver\nfox_li\n"


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


def test_published_benchmark_numbers_match_archived_csv():
    root = Path(__file__).resolve().parents[1]
    with (root / "docs" / "data" / "benchmark_operations.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        operations = {row["operation"]: row for row in csv.DictReader(stream)}
    with (root / "docs" / "data" / "fig4a_overhead_microbenchmark.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        overhead = list(csv.DictReader(stream))

    separable = operations["decompose 512^2 to order 3"]
    naive = operations["naive decompose 512^2 to order 3"]
    ratio = float(naive["best_seconds"]) / float(separable["best_seconds"])
    first = overhead[0]
    last = overhead[-1]
    ratios = [float(row["checked_to_raw_ratio"]) for row in overhead]

    expected_fragments = [
        f"{float(separable['best_seconds']):.7f}",
        f"{float(naive['best_seconds']):.7f}",
        f"{ratio / 100:.2f}",
        f"{float(first['t_raw_us']):.2f}",
        f"{float(first['t_checked_us']):.2f}",
        f"{float(last['t_raw_us']) / 1000:.3f}",
        f"{float(last['t_checked_us']) / 1000:.3f}",
        f"{min(ratios):.3f}",
        f"{max(ratios):.3f}",
    ]
    for path in [
        root / "README.md",
        root / "paper.md",
        root / "docs" / "PERFORMANCE.md",
        root / "docs" / "paper_cpc.tex",
    ]:
        content = path.read_text(encoding="utf-8")
        for fragment in expected_fragments:
            assert fragment in content, f"{path.name} is missing benchmark value {fragment}"


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


def test_spectral_centroid_is_zero_for_a_symmetric_spectrum():
    module = _load_figure_generator()
    angular_frequency = np.array([-2.0, -1.0, 1.0, 2.0])
    spectrum = np.array([1.0, 3.0, 3.0, 1.0])

    assert module._spectral_centroid(angular_frequency, spectrum) == pytest.approx(0.0)


def test_spectral_centroid_preserves_frequency_shift_direction():
    module = _load_figure_generator()
    angular_frequency = np.array([-1.0, 0.0, 1.0])
    spectrum = np.array([1.0, 1.0, 4.0])

    assert module._spectral_centroid(angular_frequency, spectrum) > 0.0


def test_wavelength_equivalent_offset_uses_declared_envelope_sign():
    module = _load_figure_generator()

    positive = module._wavelength_equivalent_offset_nm(1.0e12, 1550.0e-9)
    negative = module._wavelength_equivalent_offset_nm(-1.0e12, 1550.0e-9)

    assert positive > 0.0
    assert negative < 0.0
    assert positive == pytest.approx(-negative)


def test_alignment_negative_control_is_a_quadrature_metric_defect():
    module = _load_figure_generator()
    forward, _, _, adjoint_at = module._alignment_problem()
    theta = 200e-6
    correct = adjoint_at(theta, use_quadrature_metric=True)
    flawed = adjoint_at(theta, use_quadrature_metric=False)

    correct_report = module.dot_test(
        forward, correct, np.array([theta]), seed=0, eps=1e-7
    )
    flawed_report = module.dot_test(
        forward, flawed, np.array([theta]), seed=0, eps=1e-7
    )
    assert correct_report["rel_error"] < 1e-5
    assert flawed_report["rel_error"] > 1e-2

    probe_a = np.linspace(0.5, 1.5, 256).astype(complex)
    probe_b = np.linspace(1.5, 0.5, 256).astype(complex) * (1.0 + 0.3j)
    ratio_a = flawed(probe_a)[0] / correct(probe_a)[0]
    ratio_b = flawed(probe_b)[0] / correct(probe_b)[0]
    assert ratio_a != pytest.approx(ratio_b, rel=1e-3)


def test_alignment_documentation_describes_the_quadrature_metric_defect():
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "docs" / "paper_cpc.tex",
        root / "paper.md",
        root / "docs" / "RELEASE_GUIDE.md",
        root / "docs" / "generate_figures.py",
        root / "docs" / "data" / "fig4_provenance.csv",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources).lower()

    assert "factor 0.65" not in combined
    assert "factor-0.65" not in combined
    assert "scale-fault" not in combined
    assert "synthetic alignment" not in combined
    assert "uniform weights on gauss--legendre nodes" in combined


def test_figure_csv_schemas_record_measurement_definitions():
    data_dir = Path(__file__).resolve().parents[1] / "docs" / "data"

    with (data_dir / "fig4a_overhead_microbenchmark.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        fig4_fields = csv.DictReader(stream).fieldnames
    assert fig4_fields is not None
    assert "checked_to_raw_ratio" in fig4_fields
    assert "checked_minus_raw_us" in fig4_fields
    assert "overhead_ratio" not in fig4_fields

    with (data_dir / "fig6b_raman_provenance.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        provenance_fields = csv.DictReader(stream).fieldnames
    assert provenance_fields is not None
    assert {
        "variant",
        "spectral_centroid_angular_frequency_rad_s",
        "wavelength_equivalent_offset_nm",
        "centroid_definition",
        "mapping_definition",
    } <= set(provenance_fields)
