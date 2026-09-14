"""Run all optcon experiments, adjudications, and benchmarks in one pass.

Usage:
    python -m optcon.benchmarks.run_all
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

_repo_parent = str(Path(__file__).resolve().parent.parent.parent)
if _repo_parent not in sys.path:
    sys.path.insert(0, _repo_parent)


def run_section(title: str, module_path: str) -> bool:
    print("\n" + "=" * 70)
    print(f"  RUNNING: {title}")
    print(f"  Module : {module_path}")
    print("=" * 70 + "\n")
    start = time.perf_counter()
    try:
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


def main() -> int:
    print("*" * 70)
    print("  optcon: Complete Reproducibility & Benchmark Suite")
    print("*" * 70)

    stages = [
        ("End-to-End Laser Cavity Design Workflow", "optcon.examples.design_a_laser"),
        (
            "Experiment 01: Physical Invariant Guardrails & Bug Injection",
            "optcon.examples.experiment_01_guard_bench",
        ),
        (
            "Experiment 02: Cross-Engine Differential Comparison",
            "optcon.examples.experiment_02_cross_engine",
        ),
        (
            "Experiment 03: Mie Scattering Independent Reference Adjudication",
            "optcon.examples.experiment_03_mie_adjudication",
        ),
        (
            "Experiment 04: Gaussian Beam Propagation vs Analytical Closed Form",
            "optcon.examples.experiment_04_beam_adjudication",
        ),
        (
            "Experiment 05: Laser Cavity Alignment & Thermal Tolerance Budget",
            "optcon.examples.experiment_05_cavity_thermal_tolerance",
        ),
        ("Performance & Allocation Benchmarks", "optcon.benchmarks.bench"),
    ]

    results = []
    for title, module_path in stages:
        ok = run_section(title, module_path)
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
