# Performance

`optcon` uses NumPy and SciPy for its numerical kernels. FFT propagation,
modal decomposition, and temporary two-dimensional arrays account for most of
the runtime and memory use.

Run the benchmark from the repository root:

```bash
python -m optcon.benchmarks.bench
```

The command reports best-of-three wall times and peak traced allocation. It
writes the full results, Python version, and platform string to:

```text
optcon-artifacts/benchmarks/benchmark_operations.csv
```

## Modal baseline

Hermite-Gauss modes are separable:

```text
u_mn(x, y) = u_m(x) u_n(y)
```

The optimized implementation evaluates the coefficient matrix with matrix
products. The benchmark compares it with a direct implementation that builds
each two-dimensional mode explicitly. Both paths use the same public basis
function and are checked for numerical agreement before their timings are
compared.

## Solver refinement

Run the discretization checks with:

```bash
python -m optcon.benchmarks.convergence
```

Results are written to:

```text
optcon-artifacts/benchmarks/solver_convergence.csv
```

The finer solution is a numerical reference, not an analytic result. Treat the
reported errors and estimated orders as evidence for the tested configuration,
not as universal solver guarantees.

## Reading the results

- Compare runs made with the same Python, NumPy, SciPy, thread settings, and
  hardware.
- Regenerate results after changing a numerical kernel.
- Do not treat a single timing ratio as a package-level performance promise.
- Optional external engines have their own sampling and unit conventions.

There is no GPU, Torch, JAX, or reduced-precision backend in this release.
