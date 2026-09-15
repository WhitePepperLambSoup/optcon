# JOSS Readiness Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining reproducibility, modal-basis API, optional-engine, packaging, and documentation gaps identified in the latest JOSS readiness audit.

**Architecture:** Keep the closed-form optics API stable while making the modal basis explicit across decomposition, content reporting, and reconstruction. Refactor Experiment 05 around reusable numerical result records so every manuscript threshold is computed from the same ABCD/modal data used in its figures. Make CI distinguish required checks from optional engine availability, validate the wheel in an isolated environment, and remove contradictory quality claims from project documentation.

**Tech Stack:** Python 3.11+, NumPy, SciPy, pytest/pytest-cov, matplotlib, setuptools, GitHub Actions, Ruff, mypy.

## Global Constraints

- Do not modify files outside `optcon/`.
- Preserve the existing public API unless a change is explicitly additive.
- Every behavior change gets a regression test before implementation.
- Optional engines must never make the core test suite fail when unavailable.
- Published numerical claims must be generated from code, not hard-coded figure approximations.
- Do not claim tests or coverage pass until fresh commands provide evidence.

---

### Task 1: Complete the Laguerre-Gauss API contract

**Files:**
- Modify: `modes.py`
- Create: `tests/test_modes_laguerre_api.py`
- Modify: `tests/test_modes.py` only if an existing assertion must be updated

**Interfaces:**
- `mode_content(field, waist, max_order=3, *, radius_of_curvature=None, basis="hermite") -> list[tuple[tuple[int, int], float]]`
- `reconstruct(coefficients, field, waist, *, radius_of_curvature=None, basis="hermite") -> Field`
- For `basis="laguerre"`, coefficient keys are `(p, ell)`, with `p >= 0` and integer `ell`; negative `ell` is a valid azimuthal charge, not an array index.

- [ ] **Step 1: Write failing tests** for LG decomposition → `mode_content` → reconstruction, negative-charge reconstruction, and invalid basis.
- [ ] **Step 2: Run `python -m pytest tests/test_modes_laguerre_api.py -q` and confirm the failure is the missing `basis` API / incorrect reconstruction basis.
- [ ] **Step 3: Add the smallest implementation: thread `basis` through `mode_content` and `reconstruct`, use the generic basis path for LG, validate coefficient indices, and keep the HG separable path unchanged.
- [ ] **Step 4: Run focused tests, then `python -m pytest tests/test_modes.py tests/test_modes_laguerre_api.py -q`.
- [ ] **Step 5: Refactor docstrings and add the LG API to the generated module documentation.

### Task 2: Make Experiment 05 scientifically self-auditing

**Files:**
- Modify: `examples/experiment_05_cavity_thermal_tolerance.py`
- Create: `tests/test_experiment_05.py`
- Modify: `paper.md`

**Interfaces:**
- Add a reusable `ThermalPoint` or equivalent internal result structure for one thermal-lens power.
- `run_experiment()` returns scalar metrics including `tilt_90pct_threshold_urad`, `thermal_90pct_negative_limit_d_m`, `thermal_90pct_positive_limit_d_m`, `min_thermal_coupling`, and `max_tilt_discrepancy`.

- [ ] **Step 1: Write failing tests** asserting the result contains computed tolerance metrics, the 90% tilt threshold is finite and near 0.64 mrad, and the thermal limits are derived from the sampled coupling curve.
- [ ] **Step 2: Run `python -m pytest tests/test_experiment_05.py -q` and confirm failure because the current result only exposes three scalars.
- [ ] **Step 3: Extract thermal ABCD calculation into a helper and derive all threshold metrics from the returned arrays.
- [ ] **Step 4: Build the 2-D tolerance map by interpolating the actual thermal sweep, not an independent hard-coded approximation.
- [ ] **Step 5: Run the focused experiment test and `python -m optcon.examples.experiment_05_cavity_thermal_tolerance`.
- [ ] **Step 6: Update manuscript values and wording to match the generated metrics exactly.

### Task 3: Repair optional-engine compatibility and CI semantics

**Files:**
- Modify: `engines/registry.py`
- Modify: `engines/mie.py` if the compatibility shim belongs at the adapter boundary
- Modify: `.github/workflows/ci.yml`
- Create: `tests/test_engine_status.py`

**Interfaces:**
- `import_engine("PyMieScatt")` should return a truthful status and include the compatibility reason only when a legacy SciPy symbol is patched.
- CI must report optional engine availability separately from required core pass/fail, and must not describe skipped engines as validated.

- [ ] **Step 1: Write a failing regression test for PyMieScatt import under SciPy versions without `scipy.integrate.trapz`.
- [ ] **Step 2: Run the focused test and capture the current ImportError.
- [ ] **Step 3: Add a narrowly scoped compatibility shim before importing PyMieScatt, without changing core SciPy behavior outside that import path.
- [ ] **Step 4: Add an engine-status report step to CI and make the extended-engine job emit PASS/SKIP/FAIL counts.
- [ ] **Step 5: Run registry and engine tests locally; verify missing engines remain SKIP rather than false PASS.

### Task 4: Fix packaging and clean-environment verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Create: `tests/test_packaging_metadata.py`

**Interfaces:**
- Replace the recursive `all = ["optcon[dev,engines]"]` extra with an explicit dependency list.
- Ensure installed wheel verification covers `optcon.benchmarks.run_all` and the Experiment 05 import without depending on repository-relative source paths.

- [ ] **Step 1: Write failing metadata tests for the `all` extra and package discovery configuration.
- [ ] **Step 2: Run the focused tests and inspect the current metadata.
- [ ] **Step 3: Make `all` explicit and add package-data/distribution checks needed by the installed package.
- [ ] **Step 4: Build a wheel and install it into a temporary clean environment; run import, benchmark help, and experiment smoke checks.
- [ ] **Step 5: Update CI package job with those exact checks.

### Task 5: Reconcile quality numbers and documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/RELEASE_GUIDE.md`
- Modify: `docs/DESIGN.md`
- Modify: `CHANGELOG.md`
- Modify: `paper.md`

**Interfaces:**
- All public documentation must distinguish collected test count from coverage and must not claim external-engine validation that CI only skipped.
- Laguerre-Gauss is documented as implemented, not roadmap-only.

- [ ] **Step 1: Add a test or script that reports the collected test count and coverage command used by CI.
- [ ] **Step 2: Run the report and replace stale 408/428 claims with generated or explicitly dated values.
- [ ] **Step 3: Correct Tier-1/Tier-2 engine wording and list current CI evidence.
- [ ] **Step 4: Update CHANGELOG and DESIGN roadmap to reflect LG support and the new reproducibility metrics.
- [ ] **Step 5: Run `ruff`, `mypy`, the full pytest suite, wheel build, and the benchmark suite.

### Task 6: JOSS manuscript render and final verification

**Files:**
- Modify: `paper.md` only for final render fixes
- Modify: `paper.bib` only when bibliography validation reports an error

- [ ] **Step 1: Render/check the JOSS manuscript with the current JOSS tooling or repository CI check.
- [ ] **Step 2: Verify every figure path, citation key, author metadata field, and statement of need.
- [ ] **Step 3: Run the complete verification matrix and record exact outputs in the release-readiness notes.
- [ ] **Step 4: Review the diff for unrelated files and avoid claiming submission readiness if any required external-engine or public-development evidence is still missing.
