"""Frozen held-out mutation campaign for three optical boundary families.

The campaign is deliberately separate from ``fault_injection``.  It uses a
frozen manifest of new, author-constructed mutations in Mie, thin-film, and
sampled-beam calculations.  The reference and mutant paths are evaluated on
the same query and compared through the package differential harness.

These cases broaden the tested failure mechanisms; they are not independent
software, blinded mutations, or a population-level sensitivity estimate.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from optcon import q
from optcon.engines.differential import compare_across_engines
from optcon.engines.reference_mie import mie_efficiencies_reference
from optcon.propagation import gaussian_field, propagate, second_moment_radius
from optcon.thinfilm import single_layer_reflectance

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "optcon-artifacts"
    / "benchmarks"
    / "heldout_mutations.csv"
)


@dataclass(frozen=True)
class HeldoutMutationDefinition:
    """One immutable mutation entry in the held-out manifest."""

    mutation_id: str
    family: str
    description: str
    is_fault: bool
    parameters: tuple[tuple[str, str], ...]
    rtol: float = 1.0e-10
    atol: float = 1.0e-12


@dataclass(frozen=True)
class HeldoutMutationResult:
    """Row-level outcome for one manifest entry."""

    mutation_id: str
    family: str
    description: str
    is_fault: bool
    reference_basis: str
    observable: str
    reference_values: str
    candidate_values: str
    max_relative_error: float
    finite_result: bool
    comparison_agrees: bool
    expected_outcome_met: bool
    diagnostic_type: str
    diagnostic_message: str
    manifest_hash: str


HELDOUT_MUTATION_MANIFEST: tuple[HeldoutMutationDefinition, ...] = (
    HeldoutMutationDefinition(
        "control-mie-harmonized-medium",
        "mie",
        "Vacuum wavelength and relative index are harmonized once for a water-medium query",
        False,
        (("medium_index", "1.33"), ("diameter_nm", "200"), ("wavelength_nm", "550")),
    ),
    HeldoutMutationDefinition(
        "fault-mie-double-medium-wavelength",
        "mie",
        "The in-medium wavelength is applied twice while the query remains finite",
        True,
        (("medium_index", "1.33"), ("diameter_nm", "200"), ("wavelength_nm", "550")),
    ),
    HeldoutMutationDefinition(
        "fault-mie-absolute-index-interpreted-as-relative",
        "mie",
        "The vacuum-relative material index is treated as already relative to the medium",
        True,
        (("medium_index", "1.33"), ("diameter_nm", "400"), ("wavelength_nm", "650")),
    ),
    HeldoutMutationDefinition(
        "fault-mie-absorption-sign",
        "mie",
        "The absorption efficiency budget is reported with the subtraction sign reversed",
        True,
        (("medium_index", "1.0"), ("diameter_nm", "300"), ("wavelength_nm", "550")),
    ),
    HeldoutMutationDefinition(
        "control-thinfilm-quarterwave-units",
        "thinfilm",
        "A quarter-wave film uses equivalent nanometre and micrometre representations",
        False,
        (("film_index", "1.8"), ("wavelength_nm", "550")),
    ),
    HeldoutMutationDefinition(
        "fault-thinfilm-missing-index-in-quarter-wave-thickness",
        "thinfilm",
        "The quarter-wave thickness omits the refractive-index factor",
        True,
        (("film_index", "1.8"), ("wavelength_nm", "550")),
    ),
    HeldoutMutationDefinition(
        "fault-thinfilm-nanometre-scale-misread",
        "thinfilm",
        "A micrometre thickness is passed as though it were nanometres",
        True,
        (("film_index", "1.8"), ("wavelength_nm", "550")),
    ),
    HeldoutMutationDefinition(
        "fault-thinfilm-film-index-substitution",
        "thinfilm",
        "The substrate index is substituted for the film index in the phase model",
        True,
        (("film_index", "1.8"), ("wavelength_nm", "550")),
    ),
    HeldoutMutationDefinition(
        "control-beam-angular-spectrum",
        "beam",
        "The sampled Gaussian field uses the declared angular-spectrum propagator",
        False,
        (("waist_um", "50"), ("wavelength_nm", "1064"), ("distance_mm", "0.2")),
    ),
    HeldoutMutationDefinition(
        "fault-beam-distance-unit-scale",
        "beam",
        "A propagation distance in millimetres is interpreted as micrometres",
        True,
        (("waist_um", "50"), ("wavelength_nm", "1064"), ("distance_mm", "0.2")),
    ),
    HeldoutMutationDefinition(
        "fault-beam-wavelength-unit-scale",
        "beam",
        "A wavelength in nanometres is interpreted as micrometres",
        True,
        (("waist_um", "50"), ("wavelength_nm", "1064"), ("distance_mm", "0.2")),
    ),
    HeldoutMutationDefinition(
        "fault-beam-fresnel-model-substitution",
        "beam",
        "A paraxial Fresnel transfer function replaces the declared angular-spectrum model",
        True,
        (("waist_um", "25"), ("wavelength_um", "10"), ("distance_mm", "0.2")),
    ),
)

_REFERENCE_BASIS = "author-constructed typed or analytic reference; not external software"


def _manifest_hash() -> str:
    payload = [asdict(definition) for definition in HELDOUT_MUTATION_MANIFEST]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


MANIFEST_HASH = _manifest_hash()


def _mie_values(
    *, m: complex, diameter_nm: float, wavelength_nm: float, medium_index: float
) -> dict[str, float]:
    return mie_efficiencies_reference(
        m, diameter_nm, wavelength_nm, n_env=medium_index
    )


def _thinfilm_values(*, film_index: float, thickness: object) -> dict[str, float]:
    reflectance = float(
        single_layer_reflectance(
            1.0,
            film_index,
            1.5,
            thickness,
            q(550.0, "nm"),
        ).value
    )
    return {"R": reflectance, "T": 1.0 - reflectance}


def _beam_values(
    *, waist: object, wavelength: object, distance: object, method: str
) -> dict[str, float]:
    field = gaussian_field(
        waist=waist,
        wavelength=wavelength,
        samples=128,
        extent=q(2.0, "mm"),
    )
    propagated = propagate(field, distance, method=method)
    radius = second_moment_radius(propagated).to_value("um")
    return {"radius_um": float(radius)}


def _case_values(mutation_id: str) -> tuple[dict[str, float], dict[str, float], str]:
    """Return reference values, candidate values, and the observable name."""
    mie_material = 1.5 + 0.01j
    if mutation_id == "control-mie-harmonized-medium":
        reference = _mie_values(
            m=mie_material, diameter_nm=200.0, wavelength_nm=550.0, medium_index=1.33
        )
        return reference, dict(reference), "Mie efficiencies"
    if mutation_id == "fault-mie-double-medium-wavelength":
        reference = _mie_values(
            m=mie_material, diameter_nm=200.0, wavelength_nm=550.0, medium_index=1.33
        )
        candidate = _mie_values(
            m=mie_material,
            diameter_nm=200.0,
            wavelength_nm=550.0 / 1.33,
            medium_index=1.33,
        )
        return reference, candidate, "Mie efficiencies"
    if mutation_id == "fault-mie-absolute-index-interpreted-as-relative":
        reference = _mie_values(
            m=mie_material, diameter_nm=400.0, wavelength_nm=650.0, medium_index=1.33
        )
        candidate = _mie_values(
            m=mie_material * 1.33,
            diameter_nm=400.0,
            wavelength_nm=650.0,
            medium_index=1.33,
        )
        return reference, candidate, "Mie efficiencies"
    if mutation_id == "fault-mie-absorption-sign":
        reference = _mie_values(
            m=mie_material, diameter_nm=300.0, wavelength_nm=550.0, medium_index=1.0
        )
        candidate = dict(reference)
        candidate["Qabs"] = -candidate["Qabs"]
        return reference, candidate, "Mie efficiencies"

    film_index = 1.8
    quarter_wave_nm = 550.0 / (4.0 * film_index)
    if mutation_id == "control-thinfilm-quarterwave-units":
        reference = _thinfilm_values(film_index=film_index, thickness=q(quarter_wave_nm, "nm"))
        candidate = _thinfilm_values(
            film_index=film_index, thickness=q(quarter_wave_nm / 1000.0, "um")
        )
        return reference, candidate, "thin-film reflectance and transmittance"
    if mutation_id == "fault-thinfilm-missing-index-in-quarter-wave-thickness":
        reference = _thinfilm_values(film_index=film_index, thickness=q(quarter_wave_nm, "nm"))
        candidate = _thinfilm_values(film_index=film_index, thickness=q(550.0 / 4.0, "nm"))
        return reference, candidate, "thin-film reflectance and transmittance"
    if mutation_id == "fault-thinfilm-nanometre-scale-misread":
        reference = _thinfilm_values(film_index=film_index, thickness=q(quarter_wave_nm, "nm"))
        candidate = _thinfilm_values(
            film_index=film_index, thickness=q(quarter_wave_nm, "um")
        )
        return reference, candidate, "thin-film reflectance and transmittance"
    if mutation_id == "fault-thinfilm-film-index-substitution":
        reference = _thinfilm_values(film_index=film_index, thickness=q(quarter_wave_nm, "nm"))
        candidate = _thinfilm_values(film_index=1.5, thickness=q(quarter_wave_nm, "nm"))
        return reference, candidate, "thin-film reflectance and transmittance"

    if mutation_id == "control-beam-angular-spectrum":
        reference = _beam_values(
            waist=q(50.0, "um"),
            wavelength=q(1064.0, "nm"),
            distance=q(0.2, "mm"),
            method="angular_spectrum",
        )
        return reference, dict(reference), "Gaussian second-moment radius"
    if mutation_id == "fault-beam-distance-unit-scale":
        reference = _beam_values(
            waist=q(50.0, "um"),
            wavelength=q(1064.0, "nm"),
            distance=q(0.2, "mm"),
            method="angular_spectrum",
        )
        candidate = _beam_values(
            waist=q(50.0, "um"),
            wavelength=q(1064.0, "nm"),
            distance=q(0.2, "um"),
            method="angular_spectrum",
        )
        return reference, candidate, "Gaussian second-moment radius"
    if mutation_id == "fault-beam-wavelength-unit-scale":
        reference = _beam_values(
            waist=q(50.0, "um"),
            wavelength=q(1064.0, "nm"),
            distance=q(0.2, "mm"),
            method="angular_spectrum",
        )
        candidate = _beam_values(
            waist=q(50.0, "um"),
            wavelength=q(1064.0, "um"),
            distance=q(0.2, "mm"),
            method="angular_spectrum",
        )
        return reference, candidate, "Gaussian second-moment radius"
    if mutation_id == "fault-beam-fresnel-model-substitution":
        reference = _beam_values(
            waist=q(25.0, "um"),
            wavelength=q(10.0, "um"),
            distance=q(0.2, "mm"),
            method="angular_spectrum",
        )
        candidate = _beam_values(
            waist=q(25.0, "um"),
            wavelength=q(10.0, "um"),
            distance=q(0.2, "mm"),
            method="fresnel",
        )
        return reference, candidate, "Gaussian second-moment radius"

    raise KeyError(f"unknown held-out mutation {mutation_id!r}")


def _finite(values: Mapping[str, float]) -> bool:
    return bool(values and all(np.isfinite(float(value)) for value in values.values()))


def _serialise_values(values: Mapping[str, float]) -> str:
    return json.dumps(
        {key: float(values[key]) for key in sorted(values)},
        sort_keys=True,
        separators=(",", ":"),
    )


def run_heldout_mutations() -> list[HeldoutMutationResult]:
    """Evaluate every entry in the frozen manifest exactly once."""
    results: list[HeldoutMutationResult] = []
    for definition in HELDOUT_MUTATION_MANIFEST:
        reference, candidate, observable = _case_values(definition.mutation_id)
        finite_result = _finite(reference) and _finite(candidate)

        def runner(
            engine: str,
            reference_values: Mapping[str, float] = reference,
            candidate_values: Mapping[str, float] = candidate,
        ) -> Mapping[str, float]:
            return reference_values if engine == "reference" else candidate_values

        report = compare_across_engines(
            runner,
            engines=("reference", "candidate"),
            rtol=definition.rtol,
            atol=definition.atol,
        )
        comparison_agrees = bool(report["agree"])
        expected_outcome_met = comparison_agrees is (not definition.is_fault)
        if comparison_agrees:
            diagnostic_type = "accepted-control"
            diagnostic_message = "matched reference within the frozen tolerance"
        else:
            diagnostic_type = "reference-disagreement"
            diagnostic_message = (
                f"{report['worst_key']} relative discrepancy "
                f"{report['max_rel_error']:.6e} exceeds the frozen tolerance"
            )
        results.append(
            HeldoutMutationResult(
                mutation_id=definition.mutation_id,
                family=definition.family,
                description=definition.description,
                is_fault=definition.is_fault,
                reference_basis=_REFERENCE_BASIS,
                observable=observable,
                reference_values=_serialise_values(reference),
                candidate_values=_serialise_values(candidate),
                max_relative_error=float(report["max_rel_error"]),
                finite_result=finite_result,
                comparison_agrees=comparison_agrees,
                expected_outcome_met=expected_outcome_met,
                diagnostic_type=diagnostic_type,
                diagnostic_message=diagnostic_message,
                manifest_hash=MANIFEST_HASH,
            )
        )
    return results


def summarize_heldout_mutations(
    results: list[HeldoutMutationResult],
) -> dict[str, int]:
    """Summarize the frozen campaign without interpreting it as a rate."""
    faults = [result for result in results if result.is_fault]
    controls = [result for result in results if not result.is_fault]
    return {
        "cases_total": len(results),
        "faults_total": len(faults),
        "controls_total": len(controls),
        "faults_detected": sum(not result.comparison_agrees for result in faults),
        "controls_accepted": sum(result.comparison_agrees for result in controls),
        "finite_faults": sum(result.finite_result for result in faults),
        "unexpected_outcomes": sum(not result.expected_outcome_met for result in results),
    }


def write_heldout_results(
    results: list[HeldoutMutationResult],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write row-level held-out results as an auditable CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(HeldoutMutationResult.__annotations__)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    return path


def main() -> int:
    results = run_heldout_mutations()
    summary = summarize_heldout_mutations(results)
    output = write_heldout_results(results)
    print(
        "held-out mutation campaign: "
        f"detected {summary['faults_detected']}/{summary['faults_total']} faults; "
        f"accepted {summary['controls_accepted']}/{summary['controls_total']} controls"
    )
    print(f"manifest hash: {MANIFEST_HASH}")
    print(f"wrote {output}")
    return int(summary["unexpected_outcomes"] != 0)


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "HELDOUT_MUTATION_MANIFEST",
    "MANIFEST_HASH",
    "HeldoutMutationDefinition",
    "HeldoutMutationResult",
    "run_heldout_mutations",
    "summarize_heldout_mutations",
    "write_heldout_results",
]


if __name__ == "__main__":
    raise SystemExit(main())
