"""Experiment 05 - End-to-End Scientific Case Study:
Laser Cavity Thermal Lensing & Angular Misalignment Tolerance Analysis.

This experiment provides a realistic physical research workflow:
1. Models a high-finesse Fabry-Perot cavity (Finesse ~ 3140, lambda = 1064 nm).
2. Computes higher-order Hermite-Gauss mode breakdown (TEM00 -> TEM10, TEM20)
   under micro-radian angular misalignment, comparing discrete numerical modal
   projection against analytical paraxial perturbation theory.
3. Evaluates cavity stability and mode waist distortion under intracavity
   thermal lensing via round-trip ABCD transfer matrices.
4. Generates a publication-grade 3-panel scientific figure:
   docs/figures/fig4_cavity_tolerance.png.

Run:
    python -m optcon.examples.experiment_05_cavity_thermal_tolerance
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optcon import Quantity, q, unit
from optcon.cavity import finesse, free_spectral_range, g_parameters
from optcon.elements import compose, curved_mirror, free_space, thin_lens
from optcon.modes import decompose, hermite_gauss
from optcon.propagation import Field


def run_experiment() -> dict[str, float]:
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
    print("\n--- Sweeping Thermal Lens Power D_th (-35 to +15 m^-1) ---")
    diopters = np.linspace(-35.0, 15.0, 51)
    g_prod_thermal = []
    waist_thermal_um = []
    coupling_thermal = []

    # Nominal waist position is at z = L/2 (symmetric cavity)
    # Thin thermal lens at cavity center: f_th = 1 / D
    for d in diopters:
        if abs(d) < 1e-4:
            f_th = 1e9
        else:
            f_th = 1.0 / d

        # Round-trip ABCD from cavity center in beam order:
        # center -> space(L/2) -> mirror(R) -> space(L/2) -> center(lens) -> space(L/2) -> mirror(R) -> space(L/2) -> center
        space = free_space(q(length_m / 2.0, "m"))
        mirror = curved_mirror(radius_c)
        lens = thin_lens(q(f_th, "m"))
        round_trip_tm = compose(space, mirror, space, lens, space, mirror, space)
        round_trip = round_trip_tm.matrix

        # Verify physical passivity: determinant of ABCD ray matrix must equal unity
        det = np.linalg.det(round_trip)
        assert abs(det - 1.0) < 1e-6, f"Symplectic energy violation: det(M) = {det}"

        a = round_trip[0, 0]
        b = round_trip[0, 1]
        d_elem = round_trip[1, 1]

        # Stability parameter: m = (A + D) / 2, stable if -1 <= m <= 1
        m_val = (a + d_elem) / 2.0
        # g1*g2 = (m + 1) / 2
        g_eff = (m_val + 1.0) / 2.0
        g_prod_thermal.append(g_eff)

        if abs(m_val) < 0.999 and abs(b) > 1e-9 and g_eff > 0.0:
            # Eigenmode spot size: w_c^2 = (lambda |B|) / (pi sqrt(1 - m^2))
            wc_sq = (wavelength_m * abs(b)) / (math.pi * math.sqrt(1.0 - m_val**2))
            wc = math.sqrt(wc_sq)
            waist_thermal_um.append(wc * 1e6)
            # Overlap efficiency between nominal w0 and thermal wc:
            # eta = 4 / (w0/wc + wc/w0)^2
            eta = 4.0 / ((w0_m / wc + wc / w0_m) ** 2)
            coupling_thermal.append(eta)
        else:
            waist_thermal_um.append(np.nan)
            coupling_thermal.append(0.0)

    # 4. Generate Publication-Quality Figure
    figures_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "fig4_cavity_tolerance.png"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

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
    ax1.plot(diopters, g_prod_thermal, color=color1, lw=2, label=r"Stability Product $g_1^* g_2^*$")
    ax1.axhline(0.0, color="gray", linestyle="--", alpha=0.7, label="Concentric Boundary (0.0)")
    ax1.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Flat-Flat Boundary (1.0)")
    ax1.set_xlabel(r"Thermal Lens Power $D_{\mathrm{th}} = 1/f_{\mathrm{th}}$ ($\mathrm{m}^{-1}$)", fontsize=11)
    ax1.set_ylabel(r"Effective Stability $g_1^* g_2^*$", color=color1, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(diopters, waist_thermal_um, color=color2, lw=2, linestyle="-.", label=r"Waist $w_c$ ($\mu\mathrm{m}$)")
    ax1_twin.set_ylabel(r"Cavity Waist $w_c$ ($\mu\mathrm{m}$)", color=color2, fontsize=11)
    ax1_twin.tick_params(axis="y", labelcolor=color2)
    ax1.set_title("(b) Stability & Waist vs. Thermal Lens", fontsize=12, fontweight="bold")

    # Panel (c): 2D Tolerance Map
    ax2 = axes[2]
    tilt_mesh, d_mesh = np.meshgrid(np.linspace(0.0, 2500.0, 60), np.linspace(-30.0, 10.0, 60))
    # Combined coupling: eta_tilt * eta_thermal
    tilt_loss = np.exp(-((tilt_mesh * 1e-6) / theta_div_rad) ** 2)
    # Physical thermal mismatch efficiency
    m_arr = -0.5 - d_mesh * (length_m / 2.0 * (1.0 - length_m / 0.20))
    # Stable mask
    stable_mask = (m_arr > -1.0) & (m_arr < 1.0)
    b_const = length_m * (1.0 - length_m / (4.0 * 0.10))
    wc_mesh = np.sqrt(np.where(stable_mask, (wavelength_m * abs(b_const)) / (math.pi * np.sqrt(np.clip(1.0 - m_arr**2, 1e-6, 1.0))), np.nan))
    thermal_loss = np.where(stable_mask, 4.0 / ((w0_m / wc_mesh + wc_mesh / w0_m) ** 2), 0.0)
    eta_total = tilt_loss * np.clip(thermal_loss, 0.0, 1.0)

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
    plt.close(fig)
    print(f"\n[OK] Publication-grade figure saved to: {fig_path}")

    return {
        "max_tilt_discrepancy": max_tilt_discrepancy,
        "nominal_waist_um": w0_m * 1e6,
        "finesse": cav_finesse,
    }


def main() -> int:
    res = run_experiment()
    print("\n" + "#" * 72)
    print(f"  Experiment 05 Completed Successfully! Discrepancy: {res['max_tilt_discrepancy']:.2e}")
    print("#" * 72)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
