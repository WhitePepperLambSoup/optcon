"""Run all optcon experiments, adjudications, and benchmarks in one pass.

Usage:
    python -m optcon.benchmarks.run_all
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

from optcon.engines import available_engines

_repo_parent = str(Path(__file__).resolve().parent.parent.parent)
if _repo_parent not in sys.path:
    sys.path.insert(0, _repo_parent)


STAGES = (
    ("End-to-End Laser Cavity Design Workflow", "optcon.examples.design_a_laser"),
    (
        "Experiment 01: Physical Invariant Guardrails & Bug Injection",
        "optcon.examples.experiment_01_guard_bench",
    ),
    (
        "Semantic-contract fault-injection corpus",
        "optcon.benchmarks.fault_injection",
    ),
    (
        "Frozen held-out mutation campaign",
        "optcon.benchmarks.heldout_mutations",
    ),
    (
        "Independent compositional promotion protocol",
        "optcon.benchmarks.compositional_protocol",
    ),
    (
        "Multi-parameter adjoint stress sweep",
        "optcon.benchmarks.adjoint_stress",
    ),
    (
        "Experiment 02: Cross-Engine Differential Comparison",
        "optcon.examples.experiment_02_cross_engine",
    ),
    (
        "Experiment 03: Mie Scattering Matched-Reference Adjudication",
        "optcon.examples.experiment_03_mie_adjudication",
    ),
    (
        "Thin-film closed-form and TMM adjudication",
        "optcon.benchmarks.thinfilm_adjudication",
    ),
    (
        "Experiment 07: Thin-film stack decision impact",
        "optcon.examples.experiment_07_thinfilm_decision_impact",
    ),
    (
        "Experiment 04: Gaussian Beam Propagation vs Analytical Closed Form",
        "optcon.examples.experiment_04_beam_adjudication",
    ),
    (
        "Experiment 05: Laser Cavity Alignment & Thermal Tolerance Budget",
        "optcon.examples.experiment_05_cavity_thermal_tolerance",
    ),
    (
        "Solver discretization-convergence measurements",
        "optcon.benchmarks.convergence",
    ),
    ("Performance & Allocation Benchmarks", "optcon.benchmarks.bench"),
)


OPTIONAL_STAGE_REQUIREMENTS = {
    "optcon.benchmarks.thinfilm_adjudication": ("thin_film", "tmm_core"),
    "optcon.examples.experiment_07_thinfilm_decision_impact": ("thin_film", "tmm_core"),
}


def stage_available(module_path: str) -> bool:
    """Return whether the optional dependency required by a stage is installed."""
    requirement = OPTIONAL_STAGE_REQUIREMENTS.get(module_path)
    if requirement is None:
        return True
    group, engine = requirement
    return engine in available_engines(group)


def run_section(title: str, module_path: str, *, skip_unavailable: bool = False) -> bool:
    if skip_unavailable and not stage_available(module_path):
        print(f"[SKIP] {title}: required optional engine is unavailable")
        return True

    print("\n" + "=" * 70)
    print(f"  RUNNING: {title}")
    print(f"  Module : {module_path}")
    print("=" * 70 + "\n")
    start = time.perf_counter()
    original_argv = sys.argv
    try:
        sys.argv = [module_path]
        mod = importlib.import_module(module_path)
        if hasattr(mod, "main"):
            ret = mod.main()
            if ret is not None and ret != 0:
                print(f"[FAIL] {title} returned exit code {ret}")
                return False
        elapsed = time.perf_counter() - start
        print(f"\n[OK] {title} completed successfully in {elapsed:.2f}s")
        return True
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"\n[ERROR] {title} failed after {elapsed:.2f}s: {exc}")
        return False
    finally:
        sys.argv = original_argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-unavailable",
        action="store_true",
        help="skip stages whose optional engine dependencies are unavailable",
    )
    args = parser.parse_args(argv)

    print("*" * 70)
    print("  optcon: Complete Reproducibility & Benchmark Suite")
    print("*" * 70)

    results = []
    for title, module_path in STAGES:
        ok = run_section(title, module_path, skip_unavailable=args.skip_unavailable)
        results.append((title, ok))

    print("\n" + "#" * 70)
    print("  FINAL SUITE EXECUTION SUMMARY")
    print("#" * 70)
    all_passed = True
    for title, ok in results:
        status = "[PASSED]" if ok else "[FAILED]"
        if not ok:
            all_passed = False
        print(f"  {status:10s} {title}")
    print("#" * 70)

    if all_passed:
        print("  All experiments and adjudications reproduced successfully!")
        return 0
    else:
        print("  Some experiments failed or skipped.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
