# The engine registry

Every external solver known to `optcon` is described by one entry in `optcon.engines.registry.ENGINE_SPECS`. Each entry records the domain, import path, source location, length unit, angle unit, and a short convention note. The adapter layer converts at the boundary instead of relying on callers to remember a library-specific convention.

Run:

```bash
python -m optcon.benchmarks.engine_status
```

The command reports:

- `ready`: the import path and required source are available;
- `missing`: an optional dependency or source tree is absent;
- `error`: the import was attempted but failed for another reason.

`--strict` gates only engines with declared Tier-1 differential adapters. A registry entry is not, by itself, a completed cross-engine benchmark.

## Current evidence

| Domain | Engines or reference | Current evidence | Status |
| :--- | :--- | :--- | :--- |
| Mie scattering | `miepython` and `optcon_reference` | Maximum relative discrepancy `1.77e-10` over five diameters | Current measurement |
| Mie scattering | `PyMieScatt` with explicit `nMedium=1.0` and `optcon_reference` | Maximum relative discrepancy `3.06e-7` over five diameters | Current measurement |
| Mie scattering | Explicit air-medium query versus vacuum | Maximum relative discrepancy `9.87e-4` over five diameters; this is a physical-convention characterization | Current characterization |
| Gaussian beam | LightPipes `Forvard` and `Fresnel` | `Forvard` is within about `6.52e-11` relative difference at `z=zR`; `Fresnel` is 6.67--7.07% wide over the tested resolution sweep | Current measurement |
| Thin film | `tmm_core` | Fresnel interfaces, a single-layer Airy formula, and quarter-wave `(HL)^N H` Bragg stacks; 61 fixed queries with maximum `|Delta R|=2.914e-16` and `|Delta T|=9.992e-16` in the current environment | Current measurement |
| Thin film | `tmm_fast` | Conventions are registered; no numerical claim is made when the differentiable backend is unavailable | Availability-dependent |
| Ray tracing | `optiland`, `rayoptics`, `pyoptools`, `tracepy`, `optcon_paraxial` | Conventions and import status are recorded; numerical comparison requires a runnable adapter | Registry only |
| Diffraction | `poppy`, `prysm`, `diffractio`, `torchoptics`, `optcon_diffraction` | Units and adapter availability are recorded; numerical comparison requires a shared observable | Registry only |
| EM and photonic solvers | `ceviche`, `femwell`, `A_FMM`, `neuroptica`, `deeplens` | Conventions and dependency status are recorded; numerical comparison requires a shared observable | Registry only |

Missing engines are reported as unavailable, not as zero-valued measurements.

## Units and convention traps

| Engine | Boundary convention | Main risk |
| :--- | :--- | :--- |
| `tmm_core` | nanometres and radians | Mixing nm with SI metres in a stack definition |
| `tmm_fast` | SI metres and radians | Passing the same numerical values used for `tmm_core` without conversion |
| `diffractio` | micrometres internally | Its unit constants use `um=1.0` and `nm=0.001` |
| LightPipes | caller-selected consistent length unit | Grid size, wavelength, and distance must use the same unit |
| `PyMieScatt` | relative index and in-medium wavelength are normalized by the adapter before an `nMedium=1` call | Source and wheel artifacts marked `1.8.1.1` differ in how `nMedium` scales wavelength |
| `optiland` | millimetre lens data with micrometre wavelengths in some workflows | The project convention must be checked per observable |

The adapter policy is to state the physical medium, convert units explicitly, and use keyword-only physical inputs wherever argument order differs between engines.

## Adding an engine

1. Add an `EngineSpec` with its domain, module, source path, length unit, angle unit, and convention note.
2. Verify the convention from the engine's own examples or source code.
3. Define a shared observable and a physical query before adding a numerical claim.
4. Add a differential test against a closed form, an independent implementation, or another engine.
5. Let the test skip cleanly when the optional dependency is absent, while preserving an explicit registry status.
