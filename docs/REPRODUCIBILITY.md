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
python -m optcon.benchmarks.bench
python -m optcon.benchmarks.convergence
```

Generated files are placed under `optcon-artifacts/`. They are intentionally
kept out of version control because timings and optional-engine availability
depend on the machine running the commands.

Record `git rev-parse HEAD`, the Python version, and the platform string when
sharing benchmark results.
