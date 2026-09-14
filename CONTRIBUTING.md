# Contributing

## The two rules this project is built on

1. **Write the test first, and watch it fail.** A test that has never failed
   has not been shown to test anything.
2. **A wrong test gets fixed, not worked around.** If a test fails because the
   expectation was wrong about physics, correct the expectation and say so in
   the commit message.

## Where things go

| You are adding | It belongs in |
| --- | --- |
| Dimensional algebra, unit parsing | `optcon/units.py` |
| Quantity behaviour, NumPy integration | `optcon/quantity.py` |
| A physics invariant or derivative check | `optcon/checks.py` |
| Adoption ergonomics (decorators, inference) | `optcon/specs.py` |
| A new external engine | `optcon/engines/` plus an entry in `registry.py` |

## Adding an engine adapter

1. Add an `EngineSpec` to `ENGINE_SPECS` recording the domain, the module, the
   project path, and **the units that engine actually expects**. Verify the
   convention from the project's own examples - do not guess.
2. Write the adapter with keyword-only arguments. If two engines disagree
   about argument order, the adapter must make the swap impossible rather
   than document it.
3. State every physical convention explicitly, including defaults the
   upstream library supplies. A silent default is how a 0.15% error hides.
4. Add a test that skips cleanly when the engine is not importable.

## Running the checks

```bash
python -m pytest optcon/tests -q
python -m ruff check --no-cache optcon
python -m mypy optcon
python optcon/docs/gen_api_index.py          # regenerate docs/API.md
python -m optcon.benchmarks.bench            # time and memory per operation
```

The engine adapters look for their sources under the workspace root, which can
be overridden with `OPTCON_CORPUS_ROOT`. Tests for engines that are not
present skip instead of failing, so the suite is meaningful on its own.

## Reporting a disagreement between engines

That is the most useful bug report this project can receive. Please include:
the two engines, the physical inputs, the tolerance you expected, and - if one
of them disagrees with a closed-form or textbook result - which one.
