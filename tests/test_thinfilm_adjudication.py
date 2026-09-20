import csv

import pytest

from optcon.benchmarks import run_all
from optcon.benchmarks.thinfilm_adjudication import (
    DEFAULT_OUTPUT_PATH,
    run_thinfilm_adjudication,
    summarize_thinfilm_adjudication,
    write_thinfilm_adjudication,
)
from optcon.engines import available_engines

pytestmark = pytest.mark.skipif(
    "tmm_core" not in available_engines("thin_film"),
    reason="tmm_core unavailable",
)


def test_thinfilm_adjudication_covers_three_independent_formula_families():
    records = run_thinfilm_adjudication()

    assert {record.family for record in records} == {
        "interface",
        "single_layer",
        "bragg_peak",
    }
    assert len(records) == 61
    assert {record.polarization for record in records if record.family == "interface"} == {
        "s",
        "p",
    }


def test_thinfilm_adjudication_agrees_with_tmm_and_preserves_energy():
    summary = summarize_thinfilm_adjudication(run_thinfilm_adjudication())

    assert summary["records"] == 61
    assert summary["max_reflectance_abs_error"] < 1e-12
    assert summary["max_transmittance_abs_error"] < 1e-12
    assert summary["max_tmm_energy_residual"] < 1e-12


def test_thinfilm_adjudication_writes_row_level_csv(tmp_path):
    output = tmp_path / "thinfilm-adjudication.csv"
    records = run_thinfilm_adjudication()

    written = write_thinfilm_adjudication(records, output)

    assert written == output
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(records)
    assert {
        "family",
        "parameter_name",
        "parameter_value",
        "reference_reflectance",
        "tmm_reflectance",
        "reflectance_abs_error",
        "tmm_energy_residual",
    } <= set(rows[0])


def test_thinfilm_adjudication_is_part_of_the_reproducibility_suite():
    assert (
        "Thin-film closed-form and TMM adjudication",
        "optcon.benchmarks.thinfilm_adjudication",
    ) in run_all.STAGES
    assert "optcon-artifacts" in DEFAULT_OUTPUT_PATH.parts
    assert "docs" not in DEFAULT_OUTPUT_PATH.parts
