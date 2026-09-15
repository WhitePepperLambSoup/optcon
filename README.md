# optcon

Dimensional, contract-checked physical computation for optics and machine
learning.

optcon makes four things that optics code normally leaves to memory into
properties the computer can check:

* **dimensions** - `q(1064, "nm") + q(300, "mm")` is a length; adding a tilt
  in mrad to a length is an error, not a number;
* **amplitude vs power** - `T` and `sqrt(T)` are different objects, so the
  square-root class of mistake cannot pass silently;
* **physical invariants** - losslessness, passivity and reciprocity are
  declared and checked, and a waveplate, a polariser and a rotator each fail a
  different one;
* **derivatives** - a gradient is verified against finite differences and its
  adjoint with the `<Jv, w> == <v, J^T w>` dot-product test.

On top of that it carries a closed-form optics toolkit (ABCD matrices,
Gaussian beams, Fresnel, Jones calculus, cavities, fibre, gratings, thin-film
design, diffraction, beam quality, detector noise) and a layer that brings
external solvers under the same type discipline.

It is **not** a replacement for an FDTD, FEM or RCWA solver; it is the
semantic layer above them. See [docs/DESIGN.md](docs/DESIGN.md) for the full
rationale and for the two upstream discrepancies this approach has already
found.

[docs/API.md](docs/API.md) is a generated index of all public functions.

## Install

```bash
pip install -e ".[dev]"        # from the optcon directory
python -m pytest tests -q
```

The closed-form modules need only NumPy and SciPy. The engine adapters look
for their sources under the workspace root, overridable with
`OPTCON_CORPUS_ROOT`; engines that are absent make their tests skip rather
than fail.

## Quick start

```python
from optcon import q, sqrt, power_ratio, amplitude_ratio
from optcon.elements import compose, free_space, thin_lens
from optcon.gaussian import q_from_waist, beam_radius, propagate

wavelength = q(1064.0, "nm")
beam = q_from_waist(q(0.5, "mm"), wavelength)
system = compose(free_space(q(200.0, "mm")), thin_lens(q(100.0, "mm")))
focused = propagate(beam, system)

print(beam_radius(focused, wavelength).to_value("um"))

sqrt(power_ratio(0.81))          # -> 0.9, a field amplitude
sqrt(amplitude_ratio(0.9))       # -> AmplitudeOrderError: that is a sqrt too far
```

## Modules

| Module | Contents |
| --- | --- |
| `units` | dimensional algebra with angle as its own base dimension; unit-expression parser |
| `quantity` | `Quantity`: arithmetic, comparisons, NumPy ufuncs, amplitude order |
| `checks` | unitary / passive / reciprocal contracts; gradient and adjoint verification |
| `specs` | `@checked` signatures and unit inference from dataclass field names |
| `elements` | ABCD matrices, stability, effective focal length |
| `gaussian` | q parameter, beam radius, curvature, Gouy phase, mode overlap, cavity eigenmode |
| `fresnel` | Snell, critical and Brewster angles, reflectance and transmittance |
| `polarization` | Jones states and elements, Stokes parameters, extinction ratio |
| `cavity` | free spectral range, finesse, linewidth, stability, mode sizes |
| `fiber` | numerical aperture, V number, cutoff, mode field, propagation constant |
| `grating` | grating equation, dispersion, Littrow blaze angle, resolving power |
| `thinfilm` | anti-reflection design, quarter-wave stacks, Bragg mirrors |
| `diffraction` | Airy pattern, Rayleigh resolution, encircled energy, Strehl ratio |
| `beam_quality` | M squared and the beam parameter product |
| `noise` | photon energy, responsivity, shot noise, SNR, noise equivalent power |
| `laser` | threshold gain, cavity photon lifetime, slope efficiency, relaxation oscillations |
| `interferometry` | fringe visibility, coherence length, Fabry-Perot etalon |
| `aberrations` | Zernike modes (Noll indexed), Seidel decomposition, wavefront RMS |
| `radiometry` | Planck's law, Stefan-Boltzmann, Wien, etendue, photometry |
| `propagation` | a sampled `Field` and FFT propagation (angular spectrum or Fresnel) |
| `modes` | Gauss-Hermite and Laguerre-Gauss bases, modal decomposition and reconstruction |
| `mtf` | modulation transfer function, closed form and from any PSF |
| `waveguide` | guided TE modes of a symmetric slab, confinement, cutoff |
| `paraxial` | focal length of a singlet, computed by several engines |
| `thermal` | thermal lens focal length, Gaussian aperture and clipping losses |
| `nonlinear` | SHG phase matching, coherence length, quasi-phase-matching period |
| `vector_fields` | two-component fields: Stokes maps, analyzers, radial polarisation |
| `mueller` | Stokes vectors and Mueller matrices, including depolarisation |
| `engines` | registry and adapters for external solvers, plus differential testing |

## What the differential tests found

| Finding | Evidence |
| --- | --- |
| `PyMieScatt.MieQ` defaults to `nMedium=1.00027316` (air) and scales the index silently | shifts `Qext` by 1.5e-3; `nMedium=1.0` agrees with an independent Mie series to 6.4e-14 |
| LightPipes' `Fresnel` convolution propagator is 2-7% wide on a Gaussian beam | `Forvard` matches the closed form to 2.1e-6; the error does not shrink with sampling |
| Two thin-film engines with different internal units agree | `tmm_core` (nm) vs `tmm_fast` (SI metres): 4.1e-08 |
| A ray tracer and an ABCD chain agree on a thick lens | optiland's paraxial `f2` vs `optcon.elements`: 1e-06 |
| Two libraries agree on what "unitary" means | neuroptica's `is_unitary` vs `optcon.checks.is_unitary`: identical on 8 matrices |
| An external adjoint solver passes optcon's gradient verifier | ceviche's autograd `jacobian` vs finite differences: 1e-03 |
| Fresnel interface coefficients agree | `optcon.fresnel` vs `tmm_core`: 1e-12 |
| A wavefront-propagation package reproduces the Airy radius | `poppy` vs `optcon.diffraction`: 2% |
| A waveguide mode solver satisfies its own dispersion relation | every slab mode has residual < 1e-6 and `u^2 + w^2 = V^2` |
| Three independent propagators agree | optcon's FFT, `LightPipes` spectral and `diffractio` CZT: 2e-02 |

Five runnable experiments reproduce all of this (or run all in one pass):

```bash
python -m optcon.benchmarks.run_all               # reproduce all experiments & benchmarks
python -m optcon.examples.design_a_laser          # end-to-end workflow
python -m optcon.examples.experiment_01_guard_bench
python -m optcon.examples.experiment_02_cross_engine
python -m optcon.examples.experiment_03_mie_adjudication
python -m optcon.examples.experiment_04_beam_adjudication
python -m optcon.examples.experiment_05_cavity_thermal_tolerance
```

Experiment 05 writes its publication figure to `docs/figures` when run from the
source checkout. For an installed wheel, it writes to
`optcon-artifacts/figures` in the current working directory so it never needs
to modify `site-packages`. Choose another location with:

```bash
python -m optcon.examples.experiment_05_cavity_thermal_tolerance --output-dir artifacts
```
`design_a_laser` is the one to read first: it designs a two-mirror Nd:YAG
cavity using twelve modules together, and every number in its output is either
a closed-form relation or a contract that either holds or raises.

For modal work, name the basis explicitly. Hermite-Gauss keeps the separable,
low-memory path; Laguerre-Gauss supports radial index `p` and signed azimuthal
charge `ell` (including vortex modes):

```python
from optcon.modes import decompose, laguerre_gauss, reconstruct

mode = laguerre_gauss(x, y, p=1, ell=-1, waist=q(50.0, "um"))
coefficients = decompose(field, waist=q(50.0, "um"), max_order=2, basis="laguerre")
rebuilt = reconstruct(coefficients, field, waist=q(50.0, "um"), basis="laguerre")
```

## Performance

```bash
python -m optcon.benchmarks.bench
```

The benchmark reports wall time and peak allocation per operation. The modal
path was rewritten around the separability of the Hermite-Gauss basis and the
propagation transfer functions around separable phase factors and fewer
temporaries; [docs/PERFORMANCE.md](docs/PERFORMANCE.md) has the before/after
numbers, the two correctness traps the optimisation walked into, and the
sampling regime of each propagator.

| operation (512^2) | speed | peak memory |
| --- | --- | --- |
| modal decomposition | 460x faster | 16.0 -> 0.1 MiB |
| modal reconstruction | ~45x faster | 16 -> 4.1 MiB |
| Fresnel propagation | 3.7x faster | 18.0 -> 12.0 MiB |
| angular-spectrum propagation | 1.9x faster | 24.4 -> 18.4 MiB |
| MTF from a PSF | 1.9x faster | unchanged |
| field construction | 4.5x faster | 10.0 -> 6.0 MiB |


## Key Design Principles

1. **Angle as an independent base dimension**: Standard SI dimensional analysis treats radians as dimensionless, permitting angles to be added directly to bare scalars. In optical alignment and beam propagation, this frequently masks unit errors. `optcon` defines `angle` as an independent base dimension alongside length, mass, and time.
2. **Amplitude order algebra (`amp_order`)**:
   Every `Quantity` optionally carries an integer amplitude order $k$:
   - `None`: Untracked / permissive numeric mode (default)
   - `0`: Dimensionless ratio, round-trip count, or operator matrix
   - `1`: Field amplitude (electric field $E$, transmission amplitude $t = \sqrt{T}$)
   - `2`: Power or intensity (irradiance, power transmission $T = |t|^2$)

   Multiplication adds orders ($\text{order}(a \times b) = k_a + k_b$), division subtracts orders, and `sqrt` requires an even order and halves it. Adding or comparing mismatched orders raises `AmplitudeOrderError`.
3. **Boundary verification with unboxed computation**:
   Function boundaries validate dimensions and invariant contracts via `@checked`. Inner loops execute directly on unboxed NumPy arrays without wrapper overhead.
4. **Executable physical invariants and adjoint verification**:
   - `assert_unitary(M)`: Enforces energy preservation ($M^\dagger M = I$)
   - `assert_passive(M)`: Enforces passivity ($\sigma_{\max}(M) \le 1$)
   - `assert_reciprocal(M)`: Enforces reciprocity ($M = M^T$)
   - `dot_test(forward, adjoint, x)`: Claerbout's adjoint dot-product test ensuring $\langle Jv, w \rangle = \langle v, J^T w \rangle$ for gradient validation.

## Scope and Boundaries

`optcon` validates declared physical contracts and annotated dimensions. It does not replace numerical PDE solvers (FDTD, FEM, RCWA), but acts as the semantic and differential validation layer above them. Optional external solver adapters skip cleanly when their dependencies are absent.
