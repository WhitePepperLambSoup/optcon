"""Experiment 05 - End-to-End Scientific Case Study:
Laser Cavity Thermal Lensing & Angular Misalignment Tolerance Analysis.

This experiment provides a realistic physical research workflow:
1. Models a high-finesse Fabry-Perot cavity (Finesse ~ 3140, lambda = 1064 nm).
2. Computes higher-order Hermite-Gauss mode breakdown (TEM00 -> TEM10, TEM20)
   under micro-radian angular misalignment, comparing discrete numerical modal
   projection against analytical paraxial perturbation theory.
3. Evaluates cavity stability and mode waist distortion under intracavity
   thermal lensing via round-trip ABCD transfer matrices.
4. Generates a three-panel diagnostic plot in a caller-selected artifact directory.

Run:
    python -m optcon.examples.experiment_05_cavity_thermal_tolerance
    python -m optcon.examples.experiment_05_cavity_thermal_tolerance --output-dir artifacts
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Iterable, Mapping

import numpy as np
from matplotlib.figure import Figure

from optcon import (
    AMPLITUDE,
    POWER,
    EvidenceClaim,
    EvidenceRecord,
    EvidenceRequirements,
    Quantity,
    amplitude_ratio,
    evaluate_claim,
    power_ratio,
    q,
    unit,
)
from optcon.cavity import finesse, free_spectral_range, g_parameters
from optcon.elements import TransferMatrix, compose, curved_mirror, free_space
from optcon.gaussian import beam_radius, mode_matching_efficiency, self_consistent_mode
from optcon.modes import decompose, hermite_gauss
from optcon.propagation import Field
from optcon.specs import checked


@dataclass(frozen=True)
class ThermalPoint:
    """ABCD-derived response for one thermal-lens diopter value."""

    diopter_d_m: float
    stability_product: float
    beam_radius_um: float
    coupling: float
    stable: bool
    q_m: complex = complex(float("nan"), float("nan"))


def _tilted_mode_coupling(
    cavity_q_m: complex,
    *,
    nominal_waist_m: float,
    wavelength_m: float,
    tilt_rad: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Same-plane Gaussian overlap, including curvature and one-axis tilt."""
    k = 2.0 * math.pi / wavelength_m
    inverse_q = 1.0 / cavity_q_m
    radius_sq = -wavelength_m / (math.pi * inverse_q.imag)
    exponent = 1.0 / nominal_waist_m**2 + 1.0 / radius_sq - 0.5j * k * inverse_q.real
    nominal_q = q(1j * math.pi * nominal_waist_m**2 / wavelength_m, "m")
    aligned = mode_matching_efficiency(nominal_q, q(cavity_q_m, "m"))
    attenuation = np.exp(-0.5 * (k * np.asarray(tilt_rad)) ** 2 * (1.0 / exponent).real)
    return np.asarray(np.clip(aligned * attenuation, 0.0, 1.0))


def _thermal_point(
    diopter_d_m: float,
    *,
    length_m: float,
    radius_c: Quantity,
    wavelength_m: float,
    nominal_waist_m: float,
) -> ThermalPoint:
    """Evaluate one thermal lens directly from its round-trip ABCD matrix."""
    half_space = free_space(q(length_m / 2.0, "m"))
    mirror = curved_mirror(radius_c)
    lens = TransferMatrix(
        np.array([[1.0, 0.0], [-diopter_d_m, 1.0]]), unit("m"), "thermal_lens"
    )
    # Reference plane: immediately after the midpoint lens. Each half-trip
    # returns through that lens, so a full round trip contains two crossings.
    half_trip = compose(half_space, mirror, half_space, lens)
    round_trip = (half_trip**2).matrix

    determinant = np.linalg.det(round_trip)
    if abs(determinant - 1.0) >= 1e-6:
        raise AssertionError(f"Ray-transfer determinant violation: det(M) = {determinant}")

    a = float(round_trip[0, 0])
    d_elem = float(round_trip[1, 1])
    half_trace = (a + d_elem) / 2.0
    stability_product = (half_trace + 1.0) / 2.0
    # The identical half-trips select the physical mode even when the full
    # matrix is -I (the confocal degeneracy).
    stable = abs(float(np.trace(half_trip.matrix)) / 2.0) < 1.0
    if not stable:
        return ThermalPoint(diopter_d_m, stability_product, float("nan"), 0.0, False)

    cavity_q = self_consistent_mode(half_trip)
    radius_m = beam_radius(cavity_q, q(wavelength_m, "m")).to_value("m")
    coupling = float(_tilted_mode_coupling(
        complex(cavity_q.value),
        nominal_waist_m=nominal_waist_m,
        wavelength_m=wavelength_m,
    ))
    return ThermalPoint(
        diopter_d_m, stability_product, radius_m * 1e6, coupling, True, complex(cavity_q.value)
    )


def _interpolate_crossing(
    left_x: float, left_y: float, right_x: float, right_y: float, level: float
) -> float:
    """Linearly interpolate a threshold crossing between two samples."""
    if right_y == left_y:
        return float((left_x + right_x) / 2.0)
    fraction = (level - left_y) / (right_y - left_y)
    return float(left_x + fraction * (right_x - left_x))


def _downward_threshold(axis: np.ndarray, values: np.ndarray, level: float) -> float:
    """Return the first x where a descending curve falls below ``level``."""
    for index in range(1, len(axis)):
        if values[index] < level <= values[index - 1]:
            return _interpolate_crossing(
                float(axis[index - 1]),
                float(values[index - 1]),
                float(axis[index]),
                float(values[index]),
                level,
            )
    return float(axis[-1])


@checked
def _field_overlap_to_power(
    field_overlap: Annotated[float, "1", AMPLITUDE],
) -> Annotated[float, "1", POWER]:
    """Convert a field overlap to a power coupling with provenance checked."""
    return field_overlap * field_overlap


def run_decision_impact(output_path: str | Path) -> dict[str, Any]:
    """Measure whether provenance checking blocks an invalid tolerance budget.

    The valid workflow converts a field overlap to a power coupling.  The
    fault workflow passes the already-squared power coupling into a boundary
    that requires a field amplitude.  A bare-number implementation would
    square it a second time and report a smaller, invalid tilt budget.  The
    contract rejects that path before the budget is promoted to a result.
    """
    wavelength_m = 1.064e-6
    length_m = 0.10
    radius_c = q(200.0, "mm")
    nominal_waist_m = math.sqrt(
        (wavelength_m * length_m / (2.0 * math.pi))
        * math.sqrt((2.0 * 0.20 - length_m) / length_m)
    )
    tilts_urad = np.linspace(0.0, 3000.0, 31)
    cavity_point = _thermal_point(
        0.0,
        length_m=length_m,
        radius_c=radius_c,
        wavelength_m=wavelength_m,
        nominal_waist_m=nominal_waist_m,
    )
    power_coupling = np.asarray(
        _tilted_mode_coupling(
            cavity_point.q_m,
            nominal_waist_m=nominal_waist_m,
            wavelength_m=wavelength_m,
            tilt_rad=tilts_urad * 1e-6,
        ),
        dtype=float,
    )
    field_overlap = np.sqrt(power_coupling)

    correct_power = np.asarray(
        [
            float(_field_overlap_to_power(amplitude_ratio(float(value))).to_value("1"))
            for value in field_overlap
        ],
        dtype=float,
    )
    correct_budget = _downward_threshold(tilts_urad, correct_power, level=0.9)

    fault_contract = True
    diagnostic_type = ""
    diagnostic_message = ""
    try:
        # This is the silent mistake: a power coefficient is supplied where a
        # field overlap is required.  It is finite and numerically plausible.
        _field_overlap_to_power(power_ratio(float(power_coupling[1])))
    except Exception as error:  # noqa: BLE001 - record the contract diagnostic
        fault_contract = False
        diagnostic_type = type(error).__name__
        diagnostic_message = str(error).splitlines()[0]

    evidence_claim = EvidenceClaim(
        claim_id="fabry-perot-tilt-budget",
        representation="field overlap converted to power coupling",
        observable="90 percent TEM00 tilt threshold",
        tolerance="semantic contract, finite numerical sweep, and stated paraxial scope",
        diagnostic="field-versus-power provenance contract",
        evidence_source="local contract, numerical sweep, and provenance record",
        scope="scalar paraxial symmetric Fabry-Perot cavity",
        requirements=EvidenceRequirements(),
    )
    correct_gate = evaluate_claim(
        evidence_claim,
        EvidenceRecord(
            semantic=True,
            numerical=True,
            independent_reference=False,
            provenance=True,
            scope=True,
        ),
    )
    fault_gate = evaluate_claim(
        evidence_claim,
        EvidenceRecord(
            semantic=fault_contract,
            numerical=True,
            independent_reference=False,
            provenance=True,
            scope=True,
        ),
    )

    unchecked_fault_power = power_coupling**2
    unchecked_fault_budget = _downward_threshold(
        tilts_urad,
        unchecked_fault_power,
        level=0.9,
    )
    rows = [
        {
            "case_id": "correct-field-to-power",
            "claim_id": evidence_claim.claim_id,
            "representation": evidence_claim.representation,
            "observable": evidence_claim.observable,
            "tolerance": evidence_claim.tolerance,
            "diagnostic": evidence_claim.diagnostic,
            "evidence_source": evidence_claim.evidence_source,
            "claim_scope": evidence_claim.scope,
            "contract_status": "accepted",
            "reported_budget_urad": f"{correct_budget:.12g}",
            "unchecked_candidate_budget_urad": "",
            "diagnostic_type": "",
            "diagnostic_message": "",
            "semantic_evidence": True,
            "numerical_evidence": True,
            "independent_reference_evidence": False,
            "provenance_evidence": True,
            "scope_evidence": True,
            "status": correct_gate.status,
            "required_evidence": ";".join(correct_gate.required_categories),
            "available_evidence": ";".join(correct_gate.available_categories),
            "decision_ready": correct_gate.decision_ready,
            "failed_requirements": ";".join(correct_gate.failed_requirements),
        },
        {
            "case_id": "fault-power-used-as-field",
            "claim_id": evidence_claim.claim_id,
            "representation": evidence_claim.representation,
            "observable": evidence_claim.observable,
            "tolerance": evidence_claim.tolerance,
            "diagnostic": evidence_claim.diagnostic,
            "evidence_source": evidence_claim.evidence_source,
            "claim_scope": evidence_claim.scope,
            "contract_status": "rejected",
            "reported_budget_urad": "",
            "unchecked_candidate_budget_urad": f"{unchecked_fault_budget:.12g}",
            "diagnostic_type": diagnostic_type,
            "diagnostic_message": diagnostic_message,
            "semantic_evidence": fault_contract,
            "numerical_evidence": True,
            "independent_reference_evidence": False,
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
            "reported_budget_urad",
            "unchecked_candidate_budget_urad",
            "diagnostic_type",
            "diagnostic_message",
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
        "correct_contract": True,
        "correct_budget_reported": True,
        "correct_budget_urad": correct_budget,
        "fault_contract": fault_contract,
        "fault_budget_reported": False,
        "fault_unchecked_candidate_budget_urad": unchecked_fault_budget,
        "fault_diagnostic_type": diagnostic_type,
        "fault_diagnostic_message": diagnostic_message,
        "correct_decision_ready": correct_gate.decision_ready,
        "fault_decision_ready": fault_gate.decision_ready,
    }


def _thermal_thresholds(
    diopters: np.ndarray, coupling: np.ndarray, level: float = 0.9
) -> tuple[float, float, float]:
    """Find the contiguous 90% coupling interval containing zero diopters."""
    zero = int(np.argmin(np.abs(diopters)))
    if not np.isfinite(coupling[zero]) or coupling[zero] < level:
        return float("nan"), float("nan"), float("nan")

    lower = zero
    while lower > 0 and np.isfinite(coupling[lower - 1]) and coupling[lower - 1] >= level:
        lower -= 1
    upper = zero
    while upper + 1 < len(diopters) and np.isfinite(coupling[upper + 1]) and coupling[upper + 1] >= level:
        upper += 1

    if lower == 0:
        negative = float(diopters[0])
    else:
        negative = _interpolate_crossing(
            float(diopters[lower - 1]),
            float(coupling[lower - 1]),
            float(diopters[lower]),
            float(coupling[lower]),
            level,
        )
    if upper == len(diopters) - 1:
        positive = float(diopters[-1])
    else:
        positive = _interpolate_crossing(
            float(diopters[upper]),
            float(coupling[upper]),
            float(diopters[upper + 1]),
            float(coupling[upper + 1]),
            level,
        )
    return negative, positive, min(abs(negative), positive)


def _resolve_figures_dir(output_dir: str | Path | None) -> Path:
    """Choose a writable artifact directory for the generated figure.

    ``OPTCON_OUTPUT_DIR`` and the explicit argument override the default.
    """
    if output_dir is not None:
        return Path(output_dir)
    configured = os.environ.get("OPTCON_OUTPUT_DIR")
    if configured:
        return Path(configured)
    return Path.cwd() / "optcon-artifacts" / "figures"


def _resolve_data_dir(
    figures_dir: Path, data_dir: str | Path | None
) -> Path:
    """Choose the directory for row-level data behind the figure."""
    if data_dir is not None:
        return Path(data_dir)
    return figures_dir.parent / "benchmarks"


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: tuple[str, ...],
) -> Path:
    """Write deterministic row-level data used by the cavity figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_experiment(
    output_dir: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
) -> dict[str, float]:
    print("=" * 72)
    print("  Experiment 05: Laser Cavity Alignment & Thermal Tolerance Budget")
    print("=" * 72)

    # 1. Cavity Baseline Parameters
    wavelength_m = 1.064e-6
    wavelength = q(1064.0, "nm")
    length = q(100.0, "mm")
    length_m = 0.10
    radius_c = q(200.0, "mm")
    r_mirror = 0.999  # 99.9% power reflectance

    cav_finesse = finesse(r_mirror)
    cav_fsr = free_spectral_range(length)
    g1, g2 = g_parameters(length, radius_c, radius_c)
    g_prod = g1 * g2

    # Confocal / symmetric waist: w0^2 = (lambda L / 2pi) * sqrt((2R - L) / L)
    w0_m = math.sqrt((wavelength_m * length_m / (2.0 * math.pi)) * math.sqrt((2.0 * 0.20 - length_m) / length_m))
    theta_div_rad = wavelength_m / (math.pi * w0_m)

    print(f"  Wavelength      : {wavelength}")
    print(f"  Cavity Length   : {length}")
    print(f"  Mirror Curvature: {radius_c}")
    print(f"  Finesse         : {cav_finesse:.1f}")
    print(f"  FSR             : {cav_fsr.to_value('MHz'):.2f} MHz")
    print(f"  Stability g1*g2 : {g_prod:.4f} (stable regime: 0 <= g1*g2 < 1)")
    print(f"  Eigenmode Waist : {w0_m * 1e6:.2f} um")
    print(f"  Divergence Half : {theta_div_rad * 1e3:.3f} mrad ({theta_div_rad * 1e6:.1f} urad)")

    # 2. Angular Misalignment Sweep: TEM00 -> TEM10 Leakage
    print("\n--- Sweeping Angular Tilt theta (0 to 3000 urad) ---")
    tilts_urad = np.linspace(0.0, 3000.0, 31)
    samples = 256
    extent_m = 1.6e-3  # 1.6 mm grid (~9.3 * w0)
    spacing_m = extent_m / samples

    axis = (np.arange(samples) - samples // 2) * spacing_m
    x_grid, y_grid = np.meshgrid(axis, axis)

    # Base unperturbed TEM00 field on 2D grid
    u00 = hermite_gauss(x_grid, y_grid, 0, 0, waist=Quantity(w0_m, unit("m")))

    p_tem00_num = []
    p_tem10_num = []
    p_tem20_num = []
    p_tem00_ana = []
    p_tem10_ana = []

    for t_urad in tilts_urad:
        t_rad = t_urad * 1e-6
        k = 2.0 * math.pi / wavelength_m
        # Phase tilt: exp(i * k * theta * x)
        tilted_amp = u00 * np.exp(1.0j * k * t_rad * x_grid)

        field = Field(
            tilted_amp,
            spacing=Quantity(spacing_m, unit("m")),
            wavelength=Quantity(wavelength_m, unit("m")),
        )

        coeffs = decompose(field, waist=Quantity(w0_m, unit("m")), max_order=2)
        c00 = coeffs[(0, 0)]
        c10 = coeffs[(1, 0)]
        c20 = coeffs[(2, 0)]

        p00 = abs(c00) ** 2
        p10 = abs(c10) ** 2
        p20 = abs(c20) ** 2

        # Invariant check: unitary/passive modal decomposition (sum <= 1.0)
        assert p00 + p10 + p20 <= 1.0001, "Passive contract violated in modal projection"

        p_tem00_num.append(p00)
        p_tem10_num.append(p10)
        p_tem20_num.append(p20)

        # Analytical Siegman paraxial tilt formula:
        # |c_00|^2 = exp(-(theta / theta_div)^2)
        # |c_10|^2 = (theta / theta_div)^2 * exp(-(theta / theta_div)^2)
        ratio = t_rad / theta_div_rad
        p_tem00_ana.append(math.exp(-(ratio**2)))
        p_tem10_ana.append((ratio**2) * math.exp(-(ratio**2)))

    max_tilt_discrepancy = float(np.max(np.abs(np.array(p_tem10_num) - np.array(p_tem10_ana))))
    print(f"  Max discrepancy (numerical projection vs analytical Siegman): {max_tilt_discrepancy:.3e}")

    # 3. Thermal Lensing Perturbation Sweep (Diopters D = 1/f_th)
    print("\n--- Sweeping Thermal Lens Power D_th (-35 to +45 m^-1) ---")
    diopters = np.linspace(-35.0, 45.0, 81)
    thermal_points = [
        _thermal_point(
            float(diopter),
            length_m=length_m,
            radius_c=radius_c,
            wavelength_m=wavelength_m,
            nominal_waist_m=w0_m,
        )
        for diopter in diopters
    ]
    g_prod_thermal = np.asarray(
        [point.stability_product for point in thermal_points], dtype=float
    )
    radius_thermal_um = np.asarray(
        [point.beam_radius_um for point in thermal_points], dtype=float
    )
    coupling_thermal = np.asarray(
        [point.coupling for point in thermal_points], dtype=float
    )
    thermal_90pct_negative_limit_d_m, thermal_90pct_positive_limit_d_m, thermal_90pct_budget_abs_d_m = (
        _thermal_thresholds(diopters, coupling_thermal, level=0.9)
    )
    min_thermal_coupling = float(np.min(coupling_thermal))
    tilt_90pct_threshold_urad = _downward_threshold(
        tilts_urad, np.asarray(p_tem00_num, dtype=float), level=0.9
    )
    print(f"  90% TEM00 tilt threshold: {tilt_90pct_threshold_urad:.1f} urad")
    print(
        "  90% thermal coupling interval: "
        f"[{thermal_90pct_negative_limit_d_m:.2f}, "
        f"{thermal_90pct_positive_limit_d_m:.2f}] m^-1"
    )
    print(f"  Conservative thermal budget: |D_th| < {thermal_90pct_budget_abs_d_m:.2f} m^-1")

    # 4. Generate publication-quality figure and its row-level data.
    figures_dir = _resolve_figures_dir(output_dir)
    data_dir_path = _resolve_data_dir(figures_dir, data_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tilt_rows = [
        {
            "tilt_urad": float(tilt),
            "tem00_numeric": float(tem00_num),
            "tem00_analytical": float(tem00_ana),
            "tem10_numeric": float(tem10_num),
            "tem10_analytical": float(tem10_ana),
            "tem20_numeric": float(tem20_num),
        }
        for tilt, tem00_num, tem00_ana, tem10_num, tem10_ana, tem20_num in zip(
            tilts_urad,
            p_tem00_num,
            p_tem00_ana,
            p_tem10_num,
            p_tem10_ana,
            p_tem20_num,
            strict=True,
        )
    ]
    _write_csv(
        data_dir_path / "cavity_tilt_sweep.csv",
        tilt_rows,
        (
            "tilt_urad",
            "tem00_numeric",
            "tem00_analytical",
            "tem10_numeric",
            "tem10_analytical",
            "tem20_numeric",
        ),
    )

    thermal_rows = [
        {
            "diopter_d_m": float(point.diopter_d_m),
            "stability_product": float(point.stability_product),
            "beam_radius_um": float(point.beam_radius_um),
            "coupling": float(point.coupling),
            "stable": bool(point.stable),
            "q_real_m": float(point.q_m.real),
            "q_imag_m": float(point.q_m.imag),
        }
        for point in thermal_points
    ]
    _write_csv(
        data_dir_path / "cavity_thermal_sweep.csv",
        thermal_rows,
        ("diopter_d_m", "stability_product", "beam_radius_um", "coupling", "stable", "q_real_m", "q_imag_m"),
    )

    fig_path = figures_dir / "fig4_cavity_tolerance.png"

    fig = Figure(figsize=(16, 4.5), dpi=300)
    axes = fig.subplots(1, 3)

    # Panel (a): Angular Misalignment
    ax0 = axes[0]
    ax0.plot(tilts_urad, p_tem00_num, "o", color="#1f77b4", label=r"$\mathrm{TEM}_{00}$ (Numerical)", markersize=4)
    ax0.plot(tilts_urad, p_tem00_ana, "-", color="#1f77b4", alpha=0.7, label=r"$\mathrm{TEM}_{00}$ (Siegman)")
    ax0.plot(tilts_urad, p_tem10_num, "s", color="#d62728", label=r"$\mathrm{TEM}_{10}$ (Numerical)", markersize=4)
    ax0.plot(tilts_urad, p_tem10_ana, "--", color="#d62728", alpha=0.7, label=r"$\mathrm{TEM}_{10}$ (Siegman)")
    ax0.plot(tilts_urad, p_tem20_num, "^", color="#2ca02c", label=r"$\mathrm{TEM}_{20}$ (Numerical)", markersize=4)
    ax0.axvline(theta_div_rad * 1e6, color="black", linestyle=":", label=r"Divergence $\theta_{\mathrm{div}}$")
    ax0.set_xlabel(r"Angular Misalignment $\theta$ ($\mu\mathrm{rad}$)", fontsize=11)
    ax0.set_ylabel("Normalized Modal Power", fontsize=11)
    ax0.set_title("(a) Transverse Mode Leakage vs. Tilt", fontsize=12, fontweight="bold")
    ax0.grid(True, linestyle=":", alpha=0.6)
    ax0.legend(frameon=True, fontsize=8.5, loc="center right")

    # Panel (b): Thermal Lensing Stability & Waist
    ax1 = axes[1]
    color1 = "#2b5c8f"
    color2 = "#d95f02"
    ax1.plot(diopters, g_prod_thermal, color=color1, lw=2)
    ax1.axhline(0.0, color="gray", linestyle="--", alpha=0.7)
    ax1.axhline(1.0, color="red", linestyle="--", alpha=0.7)
    ax1.set_xlabel(r"Thermal Lens Power $D_{\mathrm{th}} = 1/f_{\mathrm{th}}$ ($\mathrm{m}^{-1}$)", fontsize=11)
    ax1.set_ylabel(r"Round-trip Stability $(1+h_{\mathrm{rt}})/2$", color=color1, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(diopters, radius_thermal_um, color=color2, lw=2, linestyle="-.", label=r"Midpoint radius $w_c$ ($\mu\mathrm{m}$)")
    ax1_twin.set_ylabel(r"Midpoint Beam Radius $w_c$ ($\mu\mathrm{m}$)", color=color2, fontsize=11)
    ax1_twin.tick_params(axis="y", labelcolor=color2)
    ax1.set_title("(b) Stability & Radius vs. Thermal Lens", fontsize=12, fontweight="bold")

    # Panel (c): 2D Tolerance Map
    ax2 = axes[2]
    tilt_map_urad = np.linspace(0.0, 2500.0, 60)
    diopter_map_d_m = np.linspace(-20.0, 45.0, 60)
    tilt_mesh, d_mesh = np.meshgrid(tilt_map_urad, diopter_map_d_m)
    map_points = [
        _thermal_point(
            float(diopter), length_m=length_m, radius_c=radius_c,
            wavelength_m=wavelength_m, nominal_waist_m=w0_m,
        )
        for diopter in diopter_map_d_m
    ]
    eta_total = np.zeros_like(tilt_mesh)
    for index, point in enumerate(map_points):
        if point.stable:
            eta_total[index] = _tilted_mode_coupling(
                point.q_m, nominal_waist_m=w0_m, wavelength_m=wavelength_m,
                tilt_rad=tilt_map_urad * 1e-6,
            )

    _write_csv(
        data_dir_path / "cavity_tolerance_map.csv",
        (
            {
                "tilt_urad": float(tilt_mesh[row, column]),
                "diopter_d_m": float(d_mesh[row, column]),
                "coupling": float(eta_total[row, column]),
            }
            for row in range(eta_total.shape[0])
            for column in range(eta_total.shape[1])
        ),
        ("tilt_urad", "diopter_d_m", "coupling"),
    )

    _write_csv(
        data_dir_path / "cavity_summary.csv",
        (
            {"metric": "wavelength", "value": wavelength_m, "unit": "m"},
            {"metric": "cavity_length", "value": length_m, "unit": "m"},
            {"metric": "mirror_radius", "value": 0.20, "unit": "m"},
            {"metric": "mirror_power_reflectance", "value": r_mirror, "unit": "1"},
            {"metric": "finesse", "value": cav_finesse, "unit": "1"},
            {"metric": "nominal_waist", "value": w0_m, "unit": "m"},
            {
                "metric": "max_tilt_discrepancy",
                "value": max_tilt_discrepancy,
                "unit": "1",
            },
            {
                "metric": "tilt_90pct_threshold",
                "value": tilt_90pct_threshold_urad,
                "unit": "urad",
            },
            {
                "metric": "thermal_90pct_negative_limit",
                "value": thermal_90pct_negative_limit_d_m,
                "unit": "1/m",
            },
            {
                "metric": "thermal_90pct_positive_limit",
                "value": thermal_90pct_positive_limit_d_m,
                "unit": "1/m",
            },
            {
                "metric": "thermal_90pct_conservative_budget",
                "value": thermal_90pct_budget_abs_d_m,
                "unit": "1/m",
            },
        ),
        ("metric", "value", "unit"),
    )

    contour = ax2.contourf(tilt_mesh, d_mesh, eta_total, levels=np.linspace(0.0, 1.0, 11), cmap="viridis")
    cbar = fig.colorbar(contour, ax=ax2)
    cbar.set_label(r"Coupling Efficiency $\eta_{00}$", fontsize=11)
    # Highlight 90%, 75%, 50% tolerance contours
    lines = ax2.contour(tilt_mesh, d_mesh, eta_total, levels=[0.50, 0.75, 0.90], colors="white", linewidths=1.5)
    ax2.clabel(lines, inline=True, fontsize=8, fmt="%.2f")
    ax2.set_xlabel(r"Angular Tilt $\theta$ ($\mu\mathrm{rad}$)", fontsize=11)
    ax2.set_ylabel(r"Thermal Lens Power ($\mathrm{m}^{-1}$)", fontsize=11)
    ax2.set_title(r"(c) Combined Tolerance Budget $\eta(\theta, D_{\mathrm{th}})$", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"\n[OK] Figure saved to: {fig_path}")

    return {
        "max_tilt_discrepancy": max_tilt_discrepancy,
        "tilt_90pct_threshold_urad": tilt_90pct_threshold_urad,
        "thermal_90pct_negative_limit_d_m": thermal_90pct_negative_limit_d_m,
        "thermal_90pct_positive_limit_d_m": thermal_90pct_positive_limit_d_m,
        "thermal_90pct_budget_abs_d_m": thermal_90pct_budget_abs_d_m,
        "min_thermal_coupling": min_thermal_coupling,
        "nominal_waist_um": w0_m * 1e6,
        "finesse": cav_finesse,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for generated figures (or set OPTCON_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="directory for row-level CSV data (defaults beside the figure directory)",
    )
    args = parser.parse_args(argv)
    res = run_experiment(output_dir=args.output_dir, data_dir=args.data_dir)
    print("\n" + "#" * 72)
    print(f"  Experiment 05 Completed Successfully! Discrepancy: {res['max_tilt_discrepancy']:.2e}")
    print("#" * 72)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
