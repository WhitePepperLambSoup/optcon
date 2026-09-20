"""Adjudicate optical coating formulas against an independent TMM solver.

The benchmark compares three closed-form families with ``tmm_core``:

* Fresnel interfaces across incidence angle and polarization;
* a single lossless layer across wavelength;
* quarter-wave ``(HL)^N H`` Bragg mirrors across pair count.

Each query records reflectance, transmittance, and the external solver's
lossless energy residual. Run from the repository root with::

    python -m optcon.benchmarks.thinfilm_adjudication
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from optcon import q
from optcon.engines import available_engines
from optcon.engines.thinfilm import stack_response
from optcon.fresnel import reflectance, transmittance
from optcon.thinfilm import bragg_reflectance, single_layer_reflectance

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "optcon-artifacts"
    / "benchmarks"
    / "thinfilm_adjudication.csv"
)


@dataclass(frozen=True)
class ThinFilmAdjudicationRecord:
    """One closed-form and transfer-matrix comparison."""

    family: str
    parameter_name: str
    parameter_value: float
    polarization: str
    wavelength_nm: float
    angle_deg: float
    pairs: int
    reference_reflectance: float
    reference_transmittance: float
    tmm_reflectance: float
    tmm_transmittance: float
    reflectance_abs_error: float
    transmittance_abs_error: float
    tmm_energy_residual: float


def _record(
    *,
    family: str,
    parameter_name: str,
    parameter_value: float,
    polarization: str,
    wavelength_nm: float,
    angle_deg: float,
    pairs: int,
    reference_reflectance: float,
    reference_transmittance: float,
    n_list: Sequence[complex],
    thicknesses: Sequence[Any],
) -> ThinFilmAdjudicationRecord:
    response = stack_response(
        n_list=n_list,
        thicknesses=thicknesses,
        wavelength=q(wavelength_nm, "nm"),
        angle=q(angle_deg, "deg"),
        polarization=polarization,
        engine="tmm_core",
    )
    tmm_reflectance = float(response["R"].value)
    tmm_transmittance = float(response["T"].value)
    return ThinFilmAdjudicationRecord(
        family=family,
        parameter_name=parameter_name,
        parameter_value=float(parameter_value),
        polarization=polarization,
        wavelength_nm=float(wavelength_nm),
        angle_deg=float(angle_deg),
        pairs=pairs,
        reference_reflectance=float(reference_reflectance),
        reference_transmittance=float(reference_transmittance),
        tmm_reflectance=tmm_reflectance,
        tmm_transmittance=tmm_transmittance,
        reflectance_abs_error=abs(tmm_reflectance - reference_reflectance),
        transmittance_abs_error=abs(tmm_transmittance - reference_transmittance),
        tmm_energy_residual=abs(tmm_reflectance + tmm_transmittance - 1.0),
    )


def run_thinfilm_adjudication() -> list[ThinFilmAdjudicationRecord]:
    """Run all interface, single-layer, and ``(HL)^N H`` comparisons."""
    if "tmm_core" not in available_engines("thin_film"):
        raise RuntimeError("tmm_core is required for thin-film adjudication")

    records: list[ThinFilmAdjudicationRecord] = []
    n_air = 1.0
    n_glass = 1.5

    for angle_deg in np.linspace(0.0, 75.0, 16):
        for polarization in ("s", "p"):
            reference_r = float(
                reflectance(
                    n_air,
                    n_glass,
                    q(float(angle_deg), "deg"),
                    polarization=polarization,
                ).value
            )
            reference_t = float(
                transmittance(
                    n_air,
                    n_glass,
                    q(float(angle_deg), "deg"),
                    polarization=polarization,
                ).value
            )
            records.append(
                _record(
                    family="interface",
                    parameter_name="incidence_angle_deg",
                    parameter_value=float(angle_deg),
                    polarization=polarization,
                    wavelength_nm=550.0,
                    angle_deg=float(angle_deg),
                    pairs=0,
                    reference_reflectance=reference_r,
                    reference_transmittance=reference_t,
                    n_list=[n_air, n_glass],
                    thicknesses=[math.inf, math.inf],
                )
            )

    n_film = 1.38
    thickness_nm = 100.0
    for wavelength_nm in np.linspace(400.0, 800.0, 17):
        reference_r = float(
            single_layer_reflectance(
                n_air,
                n_film,
                n_glass,
                q(thickness_nm, "nm"),
                q(float(wavelength_nm), "nm"),
            ).value
        )
        records.append(
            _record(
                family="single_layer",
                parameter_name="wavelength_nm",
                parameter_value=float(wavelength_nm),
                polarization="s",
                wavelength_nm=float(wavelength_nm),
                angle_deg=0.0,
                pairs=0,
                reference_reflectance=reference_r,
                reference_transmittance=1.0 - reference_r,
                n_list=[n_air, n_film, n_glass],
                thicknesses=[math.inf, q(thickness_nm, "nm"), math.inf],
            )
        )

    n_high = 2.3
    n_low = 1.45
    design_wavelength_nm = 1064.0
    for pairs in range(1, 13):
        reference_r = bragg_reflectance(n_air, n_high, n_low, n_glass, pairs)
        # The closed form includes a terminal high-index quarter-wave layer.
        layers = [value for _ in range(pairs) for value in (n_high, n_low)]
        layers.append(n_high)
        thicknesses = [
            q(design_wavelength_nm / (4.0 * refractive_index), "nm")
            for refractive_index in layers
        ]
        records.append(
            _record(
                family="bragg_peak",
                parameter_name="high_low_pairs",
                parameter_value=float(pairs),
                polarization="s",
                wavelength_nm=design_wavelength_nm,
                angle_deg=0.0,
                pairs=pairs,
                reference_reflectance=reference_r,
                reference_transmittance=1.0 - reference_r,
                n_list=[n_air, *layers, n_glass],
                thicknesses=[math.inf, *thicknesses, math.inf],
            )
        )

    return records


def summarize_thinfilm_adjudication(
    records: Sequence[ThinFilmAdjudicationRecord],
) -> dict[str, int | float]:
    """Return maximum discrepancies without converting them to pass rates."""
    if not records:
        raise ValueError("records must not be empty")
    return {
        "records": len(records),
        "max_reflectance_abs_error": max(
            record.reflectance_abs_error for record in records
        ),
        "max_transmittance_abs_error": max(
            record.transmittance_abs_error for record in records
        ),
        "max_tmm_energy_residual": max(record.tmm_energy_residual for record in records),
    }


def write_thinfilm_adjudication(
    records: Sequence[ThinFilmAdjudicationRecord],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write every comparison to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ThinFilmAdjudicationRecord.__annotations__)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return path


def main() -> int:
    if "tmm_core" not in available_engines("thin_film"):
        print("thin-film adjudication skipped: tmm_core is unavailable")
        return 0
    records = run_thinfilm_adjudication()
    summary = summarize_thinfilm_adjudication(records)
    output = write_thinfilm_adjudication(records)
    print(
        "thin-film adjudication: "
        f"{summary['records']} comparisons, "
        f"max |Delta R| {summary['max_reflectance_abs_error']:.3e}, "
        f"max |Delta T| {summary['max_transmittance_abs_error']:.3e}, "
        f"max |R+T-1| {summary['max_tmm_energy_residual']:.3e}"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
