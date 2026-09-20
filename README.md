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
* **derivatives** - candidate adjoints are checked by finite-difference
  dot-product tests using either the complex Hermitian pairing or the real
  dual pairing required when real parameters produce complex fields.

On top of that it carries a closed-form optics toolkit (ABCD matrices,
Gaussian beams, Fresnel, Jones calculus, cavities, fibre, gratings, thin-film
design, diffraction, beam quality, detector noise) and a layer that brings
external solvers under the same type discipline.

It is **not** a replacement for an FDTD, FEM or RCWA solver; it is a
semantic layer that can sit above them. See [docs/DESIGN.md](docs/DESIGN.md) for the full
rationale and for two convention-sensitive comparisons that illustrate why
the semantic layer is useful.

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
| `fox_li` | Fredholm integral operator for open resonators with finite apertures, clipping loss, and tilt |
| `nlse` | G-NLSE split-step Fourier solver (SSFM) with dispersion, SPM, Raman, self-steepening, and model-appropriate conservation traces |
| `vector_fields` | two-component fields: Stokes maps, analyzers, radial polarisation |
| `mueller` | Stokes vectors and Mueller matrices, including depolarisation |
| `engines` | registry and adapters for external solvers, plus differential testing |

## Differential verification

The repository separates runnable adapter checks from registry-only engine records. The current checks cover:

| Finding | Current evidence |
| --- | --- |
| `miepython` versus the author-constructed `optcon_reference` series | Maximum relative discrepancy `1.77e-10` over five diameters. |
| PyMieScatt with explicit `nMedium=1.0` versus the author-constructed series | Maximum relative discrepancy `3.06e-7` over five diameters; the adapter states the medium explicitly. |
| Explicit air-medium versus vacuum Mie query | Maximum relative discrepancy `9.87e-4` over five diameters, showing that the stated medium changes the physical query. |
| LightPipes `Fresnel` versus the Gaussian closed form | About `6.9%` excess width at `z=z_R`, persistent from 256 to 4096 samples; `Forvard` remains within about `6.52e-11` relative difference at that point. |
| Thin-film closed forms versus registered `tmm_core` | 61 fixed interface, single-layer, and quarter-wave `(HL)^N H` stack queries; maximum absolute `R` and `T` discrepancies `2.914e-16` and `9.992e-16`. |
| Registry-only engines | The registry records conventions and availability; a numerical comparison needs an installed adapter and a shared observable. |

Optional-engine status is reported by `python -m optcon.benchmarks.engine_status` and distinguishes `ready`, `missing`, and import `error`.

The runnable examples reproduce the available checks (or run all in one pass):

```bash
python -m optcon.benchmarks.run_all               # reproduce all experiments & benchmarks
python -m optcon.benchmarks.fault_injection        # run the deterministic silent-fault corpus
python -m optcon.benchmarks.adjoint_stress         # run the multi-parameter adjoint sweep
python -m optcon.benchmarks.thinfilm_adjudication  # compare closed-form thin-film results with tmm_core
python -m optcon.examples.design_a_laser          # end-to-end workflow
python -m optcon.examples.experiment_01_guard_bench
python -m optcon.examples.experiment_02_cross_engine
python -m optcon.examples.experiment_03_mie_adjudication
python -m optcon.examples.experiment_04_beam_adjudication
python -m optcon.examples.experiment_05_cavity_thermal_tolerance
```

The cavity tolerance case study writes the figure and four row-level CSV
files: the angular sweep, thermal-lens sweep, combined tolerance map, and a
summary of the physical parameters and reported thresholds. Pass
`--output-dir` and `--data-dir` to place those artifacts elsewhere.

Experiment 05 writes its plot to `optcon-artifacts/figures` in the current
working directory. Choose another location with:

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

Run the local benchmark with:

```bash
python -m optcon.benchmarks.bench
```

The benchmark uses best-of-three wall times, reports peak traced allocation,
and writes environment provenance to
`optcon-artifacts/benchmarks/benchmark_operations.csv`. Results are local
measurements, so regenerate them before comparing machines.

Run solver refinement checks with:

```bash
python -m optcon.benchmarks.convergence
```

The command writes `optcon-artifacts/benchmarks/solver_convergence.csv`.

Run the semantic-contract fault corpus with:

```bash
python -m optcon.benchmarks.fault_injection
```

The command writes `optcon-artifacts/benchmarks/fault_injection.csv` with
row-level detector outcomes, effect magnitudes, diagnostics, and timing
provenance. The corpus is deterministic and is intended to verify the stated
fault classes, not to estimate defect prevalence in unrelated software.

The adjoint stress command writes
`optcon-artifacts/benchmarks/adjoint_stress.csv`. It evaluates a three-parameter
phase-and-amplitude map on a nonuniform quadrature grid at 24 fixed parameter
points and eight fixed probe seeds. The correct quadrature-metric adjoint and a
uniform-metric negative control are reported row by row.

## Key Design Principles

1. **Angle as an independent base dimension**: Standard SI dimensional analysis treats radians as dimensionless, permitting angles to be added directly to bare scalars. In optical alignment and beam propagation, this frequently masks unit errors. `optcon` defines `angle` as an independent base dimension alongside length, mass, and time.
2. **Amplitude order algebra (`amp_order`)**:
   Every `Quantity` optionally carries an integer amplitude order $k$:
   - `None` / $\bot$: Untracked / permissive numeric mode (default)
   - `0`: Dimensionless ratio, round-trip count, or operator matrix
   - `1`: Field amplitude (electric field $E$, transmission amplitude $t = \sqrt{T}$)
   - `2`: Power or intensity (irradiance, power transmission $T = |t|^2$)

   Multiplication adds orders ($\text{order}(a \times b) = k_a + k_b$), division subtracts orders, and `sqrt` requires an even order and halves it. Adding or comparing mismatched orders raises `AmplitudeOrderError`.
3. **Boundary verification with unboxed computation**:
   Function boundaries validate dimensions and invariant contracts via `@checked`. Inner loops execute directly on unboxed NumPy arrays; the boundary cost is measured separately and depends on workload and array size.
4. **Executable physical invariants and adjoint verification**:
   - `assert_unitary(M)`: Enforces energy preservation ($M^\dagger M = I$)
   - `assert_passive(M)`: Enforces passivity ($\sigma_{\max}(M) \le 1$)
   - `assert_reciprocal(M)`: Enforces the transpose-symmetry form of reciprocity
     ($M = M^T$) only in a declared matched reciprocal basis
   - `dot_test(forward, adjoint, x)`: uses the Hermitian identity
     $\langle Jv,w\rangle=\langle v,J^\dagger w\rangle$ for complex state
     spaces and the real dual pairing $\operatorname{Re}\langle Jv,w\rangle
     =v^T J_R^*w$ for real parameters mapped to complex fields.

## Scope and Boundaries

`optcon` validates declared physical contracts and annotated dimensions. It does not replace numerical PDE solvers (FDTD, FEM, RCWA), but acts as the semantic and differential validation layer above them. Optional external solver adapters skip cleanly when their dependencies are absent.
