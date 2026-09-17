# Release guide

This guide covers source and package releases for `optcon`.

## Prepare

1. Update the version in `pyproject.toml`, `__init__.py`, `CITATION.cff`, and
   `CHANGELOG.md`.
2. Confirm that the working tree contains no private material.
3. Run the validation commands:

```bash
python -m pytest tests -q
python -m ruff check .
python -m mypy .
python -m optcon.benchmarks.engine_status --strict
python -m build --no-isolation
git diff --check
```

## Inspect the distributions

Build output belongs in `dist/` and should contain one wheel and one source
archive. Check both before publishing:

```bash
python -m build --no-isolation
python -m zipfile -l dist/*.whl
tar -tf dist/*.tar.gz
```

The archives should contain source code, tests required by the source archive,
license files, and package metadata. They should not contain local benchmark
or build artifacts.

## Tag and publish

Create a signed or annotated version tag only after the checks pass. A GitHub
release should attach the wheel and source archive, with short notes copied
from `CHANGELOG.md`.

Publishing to a package index requires the account owner to configure that
index's trusted-publishing or token policy. Do not store publishing tokens in
the repository.
