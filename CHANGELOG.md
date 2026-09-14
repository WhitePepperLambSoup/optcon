# Changelog

All notable changes to optcon are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Modal decomposition and reconstruction now use the separability of the
  Hermite-Gauss basis instead of materialising one grid per mode: 460x faster
  and 160x less peak memory at 512^2, order 3.
- Propagation builds its transfer function from separable phase factors and
  broadcasting rather than a meshgrid, cutting peak memory by 18-25% and
  Fresnel time by 1.5x.
- FFTs moved from `numpy.fft` to `scipy.fft`, which runs the same pocketfft
  algorithm 3.8x faster for these sizes: Fresnel propagation is now 3.7x
  faster than it was and uses a third less memory.
- `py.typed` marker and `CITATION.cff` added; `docs/PERFORMANCE.md` records the
  measurements and the sampling regime of every propagator.
- `docs/API.md`, generated from the docstrings, indexes 208 public functions
  across 31 modules.

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
- Four experiments in `examples/` covering guard quality, cross-engine
  agreement, Mie adjudication and beam-propagation adjudication.
- Packaging and project infrastructure: `pyproject.toml`, `LICENSE`, `CI`,
  `CONTRIBUTING.md`, `docs/DESIGN.md`.

### Findings

- `PyMieScatt.MieQ` defaults to `nMedium=1.00027316` (air) and silently scales
  the refractive index by it, shifting `Qext` by ~0.15% relative to the
  vacuum convention used by `miepython` and by the textbook series. Stating
  the medium explicitly brings all three implementations to agreement at the
  level of 1e-13.
- LightPipes' `Fresnel` (convolution) propagator returns a Gaussian beam
  radius 2-7% wider than the closed-form `w(z)`, while `Forvard` (spectral)
  matches it to 2e-6. The offset does not shrink with sampling.

### Verified against

- `tmm_core` for the Fresnel interface coefficients, to 1e-12
- `tmm_core` for a single-layer coating, to 1e-12
- `tmm_fast` against `tmm_core` across wavelengths and angles, to 4e-08
- closed-form Gaussian beam propagation, to 2e-6
- an independently written Mie series, to 1e-13 once the medium is stated
- cavity spot sizes from the `g`-parameter formulas against the ABCD
  eigenmode, to 1e-9
