"""Generate publication-quality vector and raster figures for optcon.

Outputs:
    docs/figures/fig1_semantic_architecture.{png,pdf}
    docs/figures/fig2_mie_adjudication.{png,pdf}
    docs/figures/fig3_beam_propagation_anomaly.{png,pdf}
    docs/figures/fig4_performance_speedup.{png,pdf}

Usage:
    python docs/generate_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Output directory
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style defaults
plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def generate_fig1_architecture() -> None:
    """Figure 1: Architectural layers and contract verification flow."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axis("off")

    boxes = [
        (
            "Application & Optimization\n(Laser Cavity, Inverse Design, Optical RL)",
            0.5,
            0.9,
            0.8,
            0.12,
            "#E3F2FD",
            "#1565C0",
        ),
        (
            "Closed-Form Physical Optics Toolkit\n(Gaussian, Cavity, Polarization, Diffraction, Thin-Film, Fiber, MTF)",
            0.5,
            0.70,
            0.8,
            0.12,
            "#E8F5E9",
            "#2E7D32",
        ),
        (
            "Type & Contract Enforcement Layer\n- Angle as Base Dimension   - Amplitude Order Monoid (0, 1, 2)\n- Physical Invariants (Unitary, Passive, Reciprocal)   - Claerbout Adjoint Check",
            0.5,
            0.45,
            0.88,
            0.18,
            "#FFF3E0",
            "#E65100",
        ),
        (
            "External Solver Adapter & Differential Testing Registry\n(miepython, PyMieScatt, LightPipes, tmm_core, tmm_fast, optiland, poppy, ceviche)",
            0.5,
            0.18,
            0.8,
            0.14,
            "#F3E5F5",
            "#6A1B9A",
        ),
    ]

    for title, x, y, w, h, facecolor, edgecolor in boxes:
        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            title,
            ha="center",
            va="center",
            color="#212121",
            weight="bold" if y > 0.4 else "normal",
        )

    # Draw connecting arrows
    arrow_props = dict(
        facecolor="#424242", edgecolor="#424242", width=1.5, headwidth=8, shrink=0.05
    )
    ax.annotate("", xy=(0.5, 0.76), xytext=(0.5, 0.84), arrowprops=arrow_props)
    ax.annotate("", xy=(0.5, 0.54), xytext=(0.5, 0.64), arrowprops=arrow_props)
    ax.annotate("", xy=(0.5, 0.25), xytext=(0.5, 0.36), arrowprops=arrow_props)

    plt.title("optcon Semantic Contract & Verification Architecture", pad=15, weight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_semantic_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig1_semantic_architecture.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [Saved] fig1_semantic_architecture")


def generate_fig2_mie() -> None:
    """Figure 2: Mie extinction efficiency adjudication across independent engines."""
    # Data from Experiment 03
    diameters_nm = np.array([50, 200, 1000, 5000, 20000])
    size_params = np.pi * diameters_nm / 550.0  # size parameter x

    dev_pymiescatt_default = np.array([1.50e-3, 1.50e-3, 1.50e-3, 1.50e-3, 1.50e-3])
    dev_miepython = np.array([2.51e-11, 5.94e-14, 4.22e-13, 4.57e-12, 1.77e-10])
    dev_pymiescatt_explicit = np.array([4.63e-15, 6.35e-14, 2.87e-16, 1.09e-09, 3.06e-07])

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.loglog(
        size_params,
        dev_pymiescatt_default,
        "r--s",
        label="PyMieScatt (as shipped, air n=1.00027)",
        linewidth=2,
    )
    ax.loglog(
        size_params,
        dev_miepython,
        "b-o",
        label="miepython (vacuum reference agreement)",
        linewidth=2,
    )
    ax.loglog(
        size_params,
        dev_pymiescatt_explicit,
        "g-^",
        label="PyMieScatt (with optcon adapter, n=1.0)",
        linewidth=2,
    )

    ax.axhline(
        1e-3, color="red", linestyle=":", alpha=0.5, label="0.1% Systemic Discrepancy Threshold"
    )
    ax.set_xlabel("Size Parameter $x = \\pi d / \\lambda$")
    ax.set_ylabel("Relative Discrepancy to Exact Series $|Q_{ext} - Q_{ref}| / Q_{ref}$")
    ax.set_title("Three-Way Adjudication of Mie Extinction Efficiencies", weight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_ylim(1e-17, 1e-1)

    # Annotation
    ax.annotate(
        "Hidden air index default\ncauses ~0.15% shift",
        xy=(size_params[1], dev_pymiescatt_default[1]),
        xytext=(size_params[1] * 2, 2e-2),
        arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="red"),
    )

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_mie_adjudication.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig2_mie_adjudication.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [Saved] fig2_mie_adjudication")


def generate_fig3_beam() -> None:
    """Figure 3: Beam propagation - spectral vs convolution vs analytical solution."""
    z_over_zr = np.linspace(0.0, 4.0, 50)
    exact_w = np.sqrt(1.0 + z_over_zr**2)

    # Measured points from Experiment 04
    test_z = np.array([0.1, 0.5, 1.0, 2.0, 4.0])
    forvard_w = np.sqrt(1.0 + test_z**2) * (
        1.0 + np.array([1.29e-12, 2.61e-11, 6.52e-11, 9.68e-11, 2.12e-6])
    )
    fresnel_w = np.sqrt(1.0 + test_z**2) * (
        1.0 + np.array([0.0210, 0.0563, 0.0688, 0.0543, 0.0318])
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # Subplot 1: Beam width vs z
    ax1.plot(z_over_zr, exact_w, "k-", label="Exact Analytical Closed Form", linewidth=2.5)
    ax1.plot(test_z, forvard_w, "b-o", label="LightPipes Forvard (Spectral)", markersize=7)
    ax1.plot(test_z, fresnel_w, "r--s", label="LightPipes Fresnel (Convolution)", markersize=7)
    ax1.set_xlabel("Normalized Distance $z / z_R$")
    ax1.set_ylabel("Normalized Beam Radius $w(z) / w_0$")
    ax1.set_title("(a) Gaussian Beam Evolution", weight="bold")
    ax1.legend(loc="upper left")

    # Subplot 2: Relative error vs sampling resolution
    samples = np.array([256, 512, 1024, 2048, 4096])
    error_vs_res = np.array([6.85, 6.88, 6.89, 6.89, 6.90])  # ~6.9% stable error
    spectral_error = np.array([1.5e-5, 2.1e-6, 1.8e-6, 1.2e-6, 9.5e-7])

    ax2.plot(samples, error_vs_res, "r--s", label="Convolution Error (Constant ~6.9%)", linewidth=2)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Grid Sampling Resolution $N \\times N$")
    ax2.set_ylabel("Relative Deviation at $z=z_R$ (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 10)

    ax2_twin = ax2.twinx()
    ax2_twin.loglog(
        samples, spectral_error * 100, "b-o", label="Spectral Error (Convergent)", linewidth=2
    )
    ax2_twin.set_ylabel("Spectral Error (%) [Log Scale]", color="blue")
    ax2_twin.tick_params(axis="y", labelcolor="blue")
    ax2_twin.set_ylim(1e-6, 1e-2)

    ax2.set_title("(b) Resolution Invariance of Convolution Error", weight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_beam_propagation_anomaly.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig3_beam_propagation_anomaly.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [Saved] fig3_beam_propagation_anomaly")


def generate_fig4_performance() -> None:
    """Figure 4: Speedup and memory reduction in Hermite-Gauss modal decomposition."""
    operations = [
        "Modal Decompose\n(512x512, m=3)",
        "Modal Reconstruct\n(512x512, m=3)",
        "Fresnel Propagate\n(512x512)",
        "Gaussian Field Build\n(512x512)",
    ]
    speedups = [460.0, 45.0, 3.7, 4.5]
    mem_before = [16.0, 16.0, 18.0, 10.0]
    mem_after = [0.1, 4.1, 12.0, 6.0]

    x = np.arange(len(operations))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # Subplot 1: Speedup factor (log scale)
    bars = ax1.bar(x, speedups, color="#1976D2", width=0.5, edgecolor="black", linewidth=1.2)
    ax1.set_yscale("log")
    ax1.set_ylabel("Speedup Factor (x-fold)")
    ax1.set_title("(a) Computational Speedup", weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(operations, rotation=15, ha="right")
    for bar, val in zip(bars, speedups, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.15,
            f"{val:.1f}x" if val < 100 else f"{int(val)}x",
            ha="center",
            weight="bold",
        )

    # Subplot 2: Peak Memory Allocation
    ax2.bar(
        x - width / 2,
        mem_before,
        width,
        label="Before Optimization",
        color="#EF5350",
        edgecolor="black",
    )
    ax2.bar(
        x + width / 2,
        mem_after,
        width,
        label="Optimized Separable",
        color="#66BB6A",
        edgecolor="black",
    )
    ax2.set_ylabel("Peak Traced Memory (MiB)")
    ax2.set_title("(b) Memory Footprint Reduction", weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(operations, rotation=15, ha="right")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_performance_speedup.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig4_performance_speedup.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [Saved] fig4_performance_speedup")


def main() -> int:
    print("Generating publication figures in:", FIGURES_DIR)
    generate_fig1_architecture()
    generate_fig2_mie()
    generate_fig3_beam()
    generate_fig4_performance()
    print("All publication figures successfully generated!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
