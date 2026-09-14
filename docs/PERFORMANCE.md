# Performance and memory

The library is pure NumPy, so its cost lives in three places: FFT
propagation, the modal basis, and anything that materialises a full grid per
mode or per coefficient. This note records what was measured, what was
changed, and what the change bought.

## How to measure it yourself

```bash
python -m optcon.benchmarks.bench
```

The script reports the best wall time and the peak traced allocation for each
operation, on a 512x512 grid where applicable. Both columns matter: on a
single-core machine an optimisation that halves the time and doubles the
memory is usually a loss, because the memory is what turns a 1024x1024 run
into a swap file.

## What changed

### Modal decomposition: the separable route

A Hermite-Gauss mode is `u_m(x) u_n(y)`, and the optional wavefront curvature
is separable too, so the whole coefficient matrix is two matrix products:

```
C = dx^2 (U^H F U*)^T          U[i, m] = u_m(x_i)
```

The obvious implementation evaluates one full 2-D mode array per `(m, n)`.
For order 3 on a 512x512 grid that is sixteen 4 MiB complex arrays; the
matrix form allocates two 1-D profile arrays and a 4x4 matrix. Reconstruction
is the transpose operation, `U C U^T`.

| operation | before | after | speed | peak memory |
| --- | --- | --- | --- | --- |
| `decompose`, 512^2, order 3 | 0.277 s | 0.0006 s | **460x** | 16.0 MiB -> 0.1 MiB |
| `reconstruct`, 512^2, order 3 | ~0.05 s | 0.0011 s | ~45x | 16 MiB -> 4.1 MiB |

Getting this right required two things the tests caught immediately: the
field array is indexed `[y, x]` so the coefficient matrix comes out
transposed, and once a wavefront curvature makes the profiles complex, the
projection needs a conjugate on *both* sides. The no-curvature case hides
both mistakes, which is why the curved-beam tests are the ones that matter.

### Propagation: separable Fresnel, fewer temporaries

The Fresnel transfer function is separable,
`exp(-i pi lambda z (fx^2 + fy^2)) = exp(-i pi lambda z fx^2) exp(-i pi lambda z fy^2)`,
so it costs two 1-D exponentials instead of an `N^2` array of them. The
angular-spectrum branch still needs an elementwise square root, but it is
built by broadcasting instead of through `np.meshgrid`, and the frequency
axis is cached because it depends only on the grid.

### The FFT backend

The dominant cost of both propagators is the transform itself, and
`scipy.fft` runs the same pocketfft algorithm several times faster than the
`numpy.fft` wrapper on this build - 1.8 ms against 7.0 ms for a 512^2
complex transform. SciPy is already a dependency, so switching is free:

| transform, 512^2 complex | numpy.fft | scipy.fft |
| --- | --- | --- |
| single `fft2` | 7.0 ms | **1.8 ms** |
| `fft2` then `ifft2` | 15.0 ms | **3.9 ms** |

`workers=-1` was measured and did not help at these sizes, so the library
stays single-threaded and does not contend for cores.

| operation | before | after | speed | peak memory |
| --- | --- | --- | --- | --- |
| `propagate`, Fresnel, 512^2 | 0.028 s | 0.0076 s | 3.7x | 18.0 -> 12.0 MiB |
| `propagate`, angular spectrum, 512^2 | 0.030 s | 0.0161 s | 1.9x | 24.4 -> 18.4 MiB |
| `propagate_vector`, both components | 0.064 s | 0.0322 s | 2.0x | 28.4 -> 22.4 MiB |
| `gaussian_field`, 512^2 | 0.0050 s | 0.0011 s | 4.5x | 10.0 -> 6.0 MiB |
| `mtf_from_psf`, 512^2 | 0.012 s | 0.0062 s | 1.9x | 10.0 MiB |

The angular-spectrum branch gains less than the Fresnel one because its
transfer function has an elementwise square root that no amount of
separability removes; the memory that used to be spent on the meshgrid and
the copies is gone in both.

### MTF: one shift instead of two

`fftshift(fft2(ifftshift(x)))` equals `fftshift(fft2(x))` times an
alternating sign, so the input no longer has to be copied to move its origin
to the corner. The identity was verified numerically before it was relied on,
and the sign matrix is built once per call rather than per row.

## What was deliberately not done

* **No cache of the transfer function itself.** A 512x512 complex transfer
  function is 4 MiB; caching a few of them would trade tens of megabytes for
  a speed-up that only appears when the same distance is propagated
  repeatedly. The library caches the O(N) frequency axis instead.
* **No GPU or Torch/JAX backend.** That would change the dependency and the
  numerical contract, which is a separate decision rather than an
  optimisation.
* **No reduced-precision path.** Physical quantities already carry units and
  amplitude orders; silently dropping to float32 would undermine the
  tolerances the contract checks rely on.
* **No multithreading.** `workers=-1` was measured on both grid sizes and did
  not beat the single-threaded pocketfft, so the library avoids taking cores
  away from whatever is calling it.

## Choosing a grid, and the sampling regimes

The propagators in this library and in the engines it adapts do not share a
validity domain, and the difference is not cosmetic:

| method | needs | notes |
| --- | --- | --- |
| angular spectrum / Fresnel transfer function | `z <= extent^2 / (N lambda)` | the transfer function is undersampled beyond this; `max_propagation` reports the limit for a given field |
| `diffractio` CZT / Rayleigh-Sommerfeld | about one sample per wavelength | at 37 wavelengths per sample it returned an answer three orders of magnitude out, so the adapter refuses |
| LightPipes `Fresnel` (convolution) | no sampling escape | it is a few percent wide on a Gaussian at any sampling; use `Forvard` |

When in doubt, propagate a beam whose answer you know.
