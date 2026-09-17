# Changelog

All notable changes to optcon are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-09-17

### Added

- A correctness-matched naive two-dimensional modal-decomposition baseline and regression tests for the benchmark comparison.
- Separate one-way and round-trip Fox-Li loss fields.
- Regression coverage for negative amplitude orders, keyword-only checked arguments, complex adjoints, engine import errors, and G-NLSE self-steepening.

### Changed

- `Quantity.amp_order` now rejects negative integers; division and negative powers raise `AmplitudeOrderError` when they would create a negative order.
- `@checked` now updates `inspect.BoundArguments` in place and calls the wrapped function with `bound.args` and `bound.kwargs`, preserving positional, keyword-only, and variadic binding semantics.
- Complex adjoint checks use the Hermitian inner product via `numpy.vdot`.
- The G-NLSE nonlinear path uses fourth-order Runge-Kutta for Raman and self-steepening terms and the corrected Fourier multiplier for the configured FFT convention.
- Engine status distinguishes `ready`, `missing`, and import `error`.
- The modal benchmark reports machine-specific best-of-three measurements with environment provenance under `optcon-artifacts/benchmarks`.

## [0.1.0]

First cut. The API is usable but not frozen.

### Added

- `units.py` - dimensional algebra with angle as its own base dimension, and a
  unit-expression parser (`mW/cm^2`, `1/ps`, prefixed derived units).
- `quantity.py` - `Quantity` with arithmetic, comparisons, NumPy ufunc
  integration and an **amplitude order** that distinguishes field amplitudes
  from power ratios.
- `checks.py` - invariant contracts (`assert_unitary`, `assert_passive`,
  `assert_reciprocal`) plus derivative verification
  (`check_gradient`, `assert_adjoint`, `dot_test`).
- `specs.py` - `@checked` for annotating existing signatures, and
  `units_of_dataclass` / `to_quantities` for inferring units from field names.
- Closed-form optics, all of it typed:
  - `elements.py` - ABCD matrices, composition, stability, effective focal length
  - `gaussian.py` - q parameter, beam radius and curvature, Gouy phase, mode
    overlap, and the self-consistent cavity eigenmode
  - `fresnel.py` - Snell, critical and Brewster angles, reflectance and
    transmittance as power ratios
  - `polarization.py` - Jones states and elements, Stokes parameters,
    extinction ratio
  - `cavity.py` - free spectral range, finesse, linewidth, stability, mode sizes
  - `fiber.py` - numerical aperture, V number, cutoff, mode field, Gloge's
    propagation constant
  - `grating.py` - grating equation, angular dispersion, Littrow blaze angle,
    resolving power
  - `thinfilm.py` - anti-reflection design, quarter-wave stacks, Bragg mirrors
  - `diffraction.py` - Airy pattern, Rayleigh resolution, encircled energy,
    Strehl ratio
  - `beam_quality.py` - M squared and the beam parameter product
  - `noise.py` - photon energy, responsivity, shot noise, SNR, NEP
  - `laser.py` - threshold gain, cavity photon lifetime, slope efficiency,
    threshold pump power, relaxation oscillation frequency
  - `interferometry.py` - fringe visibility, coherence length and time, the
    Fabry-Perot etalon, two-beam contrast
  - `aberrations.py` - Noll-indexed Zernike modes, Seidel decomposition,
    wavefront RMS and PV, Strehl from coefficients
  - `radiometry.py` - Planck's law, Stefan-Boltzmann, Wien displacement,
    etendue, radiance and irradiance, photometry
  - `propagation.py` - a sampled `Field` that carries its grid, with angular
    spectrum and Fresnel transfer-function propagation
  - `modes.py` - Gauss-Hermite basis, modal decomposition and reconstruction,
    including the wavefront curvature a propagated beam acquires
  - `mtf.py` - the modulation transfer function, both as the closed form for a
    circular pupil and as the Fourier transform of any PSF
  - `waveguide.py` - guided TE modes of a symmetric slab from the
    transcendental dispersion relation, with confinement and cutoff
  - `thermal.py` - thermal lens focal length and exact Gaussian aperture
    transmission and clipping losses
  - `nonlinear.py` - second-harmonic phase mismatch, coherence length,
    quasi-phase-matching period and effective interaction length
  - `vector_fields.py` - a `VectorField` with two transverse components, Stokes
    maps, analyzer transmission and polarisation-preserving propagation
  - `mueller.py` - Stokes vectors, Mueller matrices for polarisers, retarders,
    rotators and depolarisers, with the Jones agreement checked
- `elements.refracting_surface` - curved-interface refraction, which makes
  thick-lens systems expressible and was validated against a ray tracer.
- `paraxial.py` - `thick_lens_focal_length`, the ray-tracing member of the
  adapter family: one lens description evaluated by optcon's ABCD chain or by
  optiland, and usable with `compare_across_engines`.
- The unit parser now understands parentheses, so `W/(m^2*sr)` parses, and
  solid angle joins angle as a base dimension of its own.
- `docs/ENGINES.md` - every registered engine, the units it expects, and what
  it has been validated against.
- `engines.diffractio` - a third independent propagator, with a sampling
  guard: the CZT method is refused rather than run on a grid it cannot
  integrate on, because at 37 wavelengths per sample it returned an answer
  three orders of magnitude out.
- `engines/` - adapters bringing external optical engines under the type
  layer, with every unit convention recorded as data:
  `tmm_core`, `tmm_fast`, `miepython`, `PyMieScatt`, `LightPipes`, plus
  `optcon_reference` (Mie) and `optcon_beam_reference` (closed-form Gaussian).
- `engines/differential.py` - cross-engine differential testing
  (`compare_across_engines`, `assert_engines_agree`).
- Five runnable experiments in `examples/` covering guard quality, cross-engine
  comparisons, Mie and beam adjudication, and cavity/thermal tolerance.
- Packaging and project infrastructure: `pyproject.toml`, `LICENSE`, `CI`,
  `CONTRIBUTING.md`, `docs/DESIGN.md`.

### Historical and optional findings

- The PyMieScatt default-medium experiment is retained as a conditional
  check in `examples/experiment_03_mie_adjudication.py`. It runs when the
  optional dependencies are installed.
- In the current stated finite-window configuration, LightPipes' convolution
  propagator produces a wider Gaussian radius than the closed form, while the
  spectral propagator tracks the closed form closely. The example reports the
  configuration and measured values when LightPipes is available.

### Validation coverage

- Optional third-party comparisons run when their dependencies import and a
  convention-matched observable is available.
- Closed-form and internal cross-route checks remain in the test suite. Timing
  and platform details belong to the generated benchmark output, not this
  changelog.
