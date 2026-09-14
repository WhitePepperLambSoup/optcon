# The engine registry

Every external solver optcon knows about is described by one entry in
`optcon.engines.registry.ENGINE_SPECS`. The entry records what the engine
covers, where its source lives, and **the units it actually expects**, so an
adapter can convert at the boundary instead of relying on the caller to
remember. Additions are cheap by design: one spec entry, plus one differential
test where a shared observable exists.

Engines that cannot import in a given environment report why rather than
failing obscurely, and the tests that need them skip cleanly. Point the
registry at a different corpus with `OPTCON_CORPUS_ROOT`.

## Availability and conventions

| Engine | Domain | Length | Angle | Adapter | Validated against |
| --- | --- | --- | --- | --- | --- |
| `tmm_core` | thin film | nm | rad | `stack_response` | `tmm_fast` to 4e-08 |
| `tmm_fast` | thin film | **m** | rad | `stack_response` | `tmm_core` to 4e-08 |
| `miepython` | Mie | nm | rad | `mie_efficiencies` | textbook series to 1.8e-10 |
| `PyMieScatt` | Mie | nm | rad | `mie_efficiencies` | textbook series to 1.5e-03 (its `nMedium` default; see below) |
| `optcon_reference` | Mie | nm | rad | `mie_efficiencies` | written from Bohren & Huffman |
| `lightpipes_forvard` | beam | mm | rad | `gaussian_beam_radius` | closed form to 2.1e-06 |
| `lightpipes_fresnel` | beam | mm | rad | `gaussian_beam_radius` | closed form: **2-7% wide** |
| `optcon_beam_reference` | beam | mm | rad | `gaussian_beam_radius` | `w = w0 sqrt(1 + (z/zR)^2)` |
| `optiland` | ray tracing | mm | rad | `thick_lens_focal_length` | optcon's ABCD to 1e-09 |
| `optcon_paraxial` | ray tracing | mm | rad | `thick_lens_focal_length` | thick-lens lensmaker equation |
| `rayoptics` | ray tracing | mm | rad | - | registered; `src/` layout |
| `neuroptica` | photonic circuit | um | rad | - | `is_unitary` agrees with `checks.is_unitary` |
| `ceviche` | EM solver | m | rad | - | its adjoint passes `check_gradient` |
| `femwell` | EM solver | um | rad | - | registered; its `mode_solver_1d` does not import in reasonable time here |
| `A_FMM` | EM solver | um | rad | - | registered; **unit unverified** |
| `poppy` | diffraction | m | rad | `airy_psf_radius` | closed form to 2% (it resamples the detector, so the pixel scale must be read from the FITS header, not from the request) |
| `optcon_diffraction` | diffraction | m | rad | `airy_psf_radius` | `theta = 1.22 lambda / D` |
| `prysm` | diffraction | mm | rad | - | registered; its `free_space` could not be matched to a Fresnel propagation from the docs, so no adapter ships |
| `diffractio` | beam | **um** | rad | `gaussian_beam_radius` | closed form to 5e-04 on a grid it can integrate on; **refuses** grids coarser than ~1.5 wavelengths per sample |
| `tracepy` | ray tracing | um | rad | - | its `RayGroup` docstring states microns |
| `pyoptools` | ray tracing | mm | rad | - | registered; **unit unverified** |
| `deeplens` | lens design | mm | rad | - | registered; PyTorch; **unit unverified** |
| `torchoptics` | diffraction | m | rad | - | shallow clone lacks `_version` |

A dash in the adapter column means the engine is registered and importable but
has no callable adapter yet: either no shared observable has been agreed (grid
alignment for diffraction, surface conventions for ray tracing) or the work
has not been done. The conventions are recorded anyway, because that is the
part that is expensive to rediscover.

## Why the conventions are recorded

Three of these engines compute the same physics in three different unit
systems, and none of them says so in a signature a caller would read:

| Engine | Internally consistent unit | Trap |
| --- | --- | --- |
| `tmm_core` | nanometre | none, if you stay in nm |
| `tmm_fast` | **metre (SI)** | the same stack written in nm is off by 10^9 |
| `diffractio` | **micron** | it ships `um = 1.0`, so `nm = 0.001` |
| `LightPipes` | whatever you pass | size, wavelength **and** distance must share it |
| `Meep` (surveyed) | dimensionless | `a`-units, constants set to one |
| `Tidy3D` (surveyed) | micron | a `UNIT_SCALING` dictionary |

`PyMieScatt` adds a fourth kind of trap: a *physical* default rather than a
unit one. `MieQ` ships `nMedium=1.00027316`, so a call without that argument
answers a question about a sphere in air. The adapter states the medium
explicitly for every engine, which is what closed a 0.2% disagreement that
neither library reported.

## Adding an engine

1. Add an `EngineSpec`: domain, module, project path, length unit, angle unit,
   and a one-line summary of the trap it sets.
2. Verify the convention **from the project's own examples**, not from
   documentation prose.
3. If a shared observable exists, write the adapter with keyword-only
   arguments and a differential test against the closed form or another
   engine.
4. Let the test skip when the engine is unavailable.
