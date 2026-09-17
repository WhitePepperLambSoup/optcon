# Reproducibility Record

This file records how to regenerate the measurements and figures used in the
current manuscript review. It separates reproducible procedure from
machine-specific observations.

## Review baseline

- Repository: `optcon`
- Review baseline `HEAD`: `a73dcf5`
- Review date: 2026-09-17
- Working-tree state: the review repairs are uncommitted changes on top of the
  baseline. Capture the exact final state with `git status --short` and
  `git rev-parse --verify HEAD` before creating an archive.

## Recorded environment

The current generated provenance files record:

| Item | Value |
| :--- | :--- |
| Python | 3.14.5 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| Platform | Windows 11, `10.0.26100-SP0` |

These versions describe the recorded local run. Reproducing the procedure on
another platform may produce different runtimes, floating-point last bits, or
optional-engine availability.

## Installation and validation

From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
python -m ruff check .
python -m mypy .
python -m optcon.benchmarks.engine_status --strict
python -m build --no-isolation
```

If the package is not installed in editable mode, run commands with the
repository root on `PYTHONPATH` or use the project's virtual environment. The
engine-status command reports `ready`, `missing`, and import `error`
separately. Missing optional dependencies are not silently converted into
numerical zeros.

The `dev` extra includes the PEP 517 frontend and local build backend
dependencies (`build`, `setuptools`, and `wheel`). The final command therefore
checks that an editable development environment can build both distribution
artifacts without downloading an isolated backend.

## Regenerating data and figures

```bash
python -m optcon.benchmarks.run_all
python docs/generate_figures.py
```

The first command runs the examples and benchmark stages, including solver
discretization convergence. The second command regenerates the publication
figures and their CSV traces under `docs/figures` and `docs/data`.

The principal data-to-figure mapping is:

| Figure | Data files |
| :--- | :--- |
| Figure 1 | `fig1` is generated from the architecture description in the figure script |
| Figure 2 | `fig2_mie_adjudication.csv`, `fig2_mie_provenance.csv` |
| Figure 3 | `fig3_beam_evolution.csv`, `fig3_beam_resolution.csv` |
| Figure 4 | `fig4a_overhead_microbenchmark.csv`, `fig4b_alignment_optimization.csv`, `fig4_provenance.csv` |
| Figure 5 | `fig5a_fox_li_loss.csv`, `fig5c_fox_li_tilt.csv` |
| Figure 6 | `fig6a_soliton_propagation.csv`, `fig6b_raman_spectrum.csv`, `fig6b_raman_provenance.csv`, `fig6c_energy_drift.csv` |
| Convergence table | `solver_convergence.csv` |
| Modal and operation timings | `benchmark_operations.csv` |

The Mie and beam comparisons are availability-sensitive. The provenance files
state which optional engines were importable during generation. Do not report a
result as current when its required adapter was unavailable.

In the current Windows run, `PyMieScatt` 1.8.1.1 and `shapely` 2.1.2 were
available. With the surrounding medium stated explicitly, the maximum
PyMieScatt discrepancy against the independent Mie series was `3.06e-7` over
five diameters. The library-default medium characterization reached `1.50e-3`
over the same points. The registry applies a narrowly scoped compatibility
alias for the legacy `scipy.integrate.trapz` import before loading PyMieScatt.

## Benchmark interpretation

The modal benchmark uses a `512 x 512` grid and order 3. Its timing and memory
values are observations from one recorded machine, not performance guarantees.
The current values are `0.0006501 s` versus `0.3006149 s`, with peak traced
allocations of `0.0997 MiB` versus `16.0086 MiB` for the separable and naive
implementations, respectively.
The Figure 4(a) boundary test uses one warm-up and seven timed calls, with the
median reported. At `N=10^6`, the recorded raw and checked times were about
44.046 ms and 39.309 ms; across the sampled sizes the checked-to-raw ratio ranged
from about 0.892 to 5.880. The order reversal at the largest arrays is treated as
timing variability rather than as evidence of negative wrapper cost.

The Raman+self-steepening figure uses 2048 temporal samples over a 10-ps window
and 100 longitudinal steps. This selected configuration passes the lossless
energy monitor; it does not imply that every window and step size will do so.

## Solver convergence interpretation

`solver_convergence.csv` compares candidates with finer numerical references
from the same implementation and physical configuration. The Fox-Li reference
uses 512 quadrature points. The G-NLSE references use 400 longitudinal steps,
128 temporal samples, and a 2-ps temporal window. These references are not
analytic solutions or exact continuum values. The observed G-NLSE refinement is
approximately second order over the resolved ranges; the Fox-Li sequence
reaches a machine-precision plateau above 32 points, so no universal asymptotic
order is claimed.

## Manuscript PDF

Compile `docs/paper_cpc.tex` with the local LaTeX toolchain, run BibTeX, and
compile twice more. Inspect every rendered page with a PDF renderer before
replacing the tracked PDF. Record the exact compiler versions and final
`git rev-parse --verify HEAD` alongside any archival release. A successful
build is necessary but does not replace visual inspection or a source-to-data
consistency check.
