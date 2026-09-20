# Reproducing the checks

Use Python 3.11 or newer from the repository root.

## Install

```bash
python -m pip install -e ".[dev]"
```

Install optional comparison engines when needed:

```bash
python -m pip install -e ".[dev,engines]"
```

## Validate

```bash
python -m pytest tests -q
python -m ruff check .
python -m mypy .
python -m build --no-isolation
```

Check optional engine imports separately:

```bash
python -m optcon.benchmarks.engine_status --strict
```

The status command distinguishes a missing optional dependency from an import
failure. It does not turn unavailable measurements into zeros.

## Run examples and benchmarks

```bash
python -m optcon.benchmarks.run_all
python -m optcon.benchmarks.fault_injection
python -m optcon.benchmarks.adjoint_stress
python -m optcon.benchmarks.thinfilm_adjudication
python -m optcon.benchmarks.bench
python -m optcon.benchmarks.convergence
python -m optcon.examples.experiment_05_cavity_thermal_tolerance
```

Generated files are placed under `optcon-artifacts/`. They are intentionally
kept out of version control because timings and optional-engine availability
depend on the machine running the commands.

The deterministic fault-injection command writes
`optcon-artifacts/benchmarks/fault_injection.csv`. Each row records the
injected or control case, the outcomes of the numeric-smoke, dimension-only,
and complete-contract layers, the measured effect, the diagnostic, and median
runtime. `run_all` executes the same corpus as part of the complete suite.

The adjoint stress command writes
`optcon-artifacts/benchmarks/adjoint_stress.csv`. It uses 24 deterministic
three-parameter points and eight deterministic probe seeds, for 192 row-level
comparisons. The file records both the metric-consistent result and the
uniform-metric negative control.

The thin-film adjudication command uses the installed `tmm` package or the
registered `tmm_core` source tree. It compares Fresnel interfaces, a single-layer Airy formula, and
quarter-wave `(HL)^N H` Bragg stacks at 61 fixed queries, writing
`optcon-artifacts/benchmarks/thinfilm_adjudication.csv`. If `tmm_core` is not
available, the command reports an explicit skip; it never substitutes a
placeholder result.

The cavity tolerance case study writes `fig4_cavity_tolerance.png` and four
CSV files. The CSV files contain the tilt sweep, thermal-lens sweep, combined
two-parameter tolerance map, and the scalar parameter/result summary used by
the figure. The midpoint thermal lens is crossed twice per round trip. Thermal
and joint tilt coupling use the complex Gaussian q parameters at the same
plane, including both radius and wavefront curvature. The thermal CSV records
the midpoint beam radius and both components of q. For a controlled output
location, run:

```bash
python -m optcon.examples.experiment_05_cavity_thermal_tolerance \
  --output-dir artifacts/figures \
  --data-dir artifacts/benchmarks/cavity
```

Record `git rev-parse HEAD`, the Python version, and the platform string when
sharing benchmark results.
