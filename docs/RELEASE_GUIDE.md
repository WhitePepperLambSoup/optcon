# optcon Release and Submission Guide

This guide describes the local checks and external steps needed to prepare
`optcon` for a software or computational-physics submission. Local checks are
evidence about this checkout; they do not guarantee editorial acceptance,
publication, a DOI, or a fee waiver.

## 1. Before a release candidate

Confirm the following items manually:

- The author list, affiliations, ORCID values, license, and competing-interest
  statement are correct.
- The public repository contains the source, tests, examples, documentation,
  figure sources, generated CSV data, and the manuscript files required by the
  selected venue.
- Optional-engine availability is recorded for the machine used to generate
  the reported results.
- The working tree and revision used for submission are recorded. Do not
  describe an uncommitted working tree as an immutable release.

The repository already has Git history. Do not initialize a second repository
or overwrite existing branches as part of this checklist.

## 2. Local validation gates

Run these commands from the repository root with the project environment
activated. The commands below are the validation protocol, not fixed claims
about test counts, coverage, runtime, or hardware performance.

```bash
python -m pytest tests -q
python -m ruff check .
python -m mypy .
python -m optcon.benchmarks.engine_status --strict
python -m optcon.benchmarks.run_all
python docs/generate_figures.py
git diff --check
```

`run_all` and `generate_figures.py` write benchmark and figure data. Review the
resulting CSV files, especially the environment provenance, before copying
numbers into a manuscript. Re-run the figure script after any benchmark or
solver change.

For the LaTeX manuscript, compile from `docs` with the installed TeX toolchain
and resolve all citations and references:

```bash
pdflatex paper_cpc.tex
bibtex paper_cpc
pdflatex paper_cpc.tex
pdflatex paper_cpc.tex
```

Then inspect the PDF visually after rendering every page to PNG. Check for
clipped text, bad equation breaks, unreadable legends, table overflow, missing
glyphs, and undefined references. A successful compiler exit code alone is not
enough.

## 3. Manuscript packages

`paper.md` is the JOSS-style manuscript source. `docs/paper_cpc.tex` is the
Elsevier/CPC-oriented LaTeX source. They have different format and submission
requirements, so validate the source against the current author guidelines of
the venue actually selected.

Before submission, verify that:

- every reported measurement can be traced to a CSV row or a documented test;
- optional or unavailable engines are clearly labeled and are not represented
  by fabricated data;
- the controlled nonuniform-grid alignment example is not described as a complete inverse-
  design, automatic-differentiation, or reinforcement-learning system;
- numerical references used for convergence are identified as finer numerical
  discretizations rather than analytic truth;
- limitations, external-engine conventions, and hardware dependence are
  visible to reviewers;
- the AI-use disclosure matches the authors' actual process.

## 4. Git and archival workflow

Use the existing repository workflow chosen by the authors. Before publishing
an immutable release, record the exact commit and verify the generated archive
contains the intended source and figures. Create a tag only when the authors
and venue process call for one, and push it only after reviewing the tag and
remote.

Do not invent a tag, commit, DOI, archive identifier, or acceptance status in
the manuscript or release notes. A DOI is issued by an external archive or
publisher only after the corresponding external action succeeds.

## 5. External submission steps

1. Choose the target venue and read its current author, software, data, and
   AI-disclosure requirements.
2. Submit the appropriate manuscript source and repository revision through
   the venue's official system.
3. Respond to editorial and reviewer requests using new, reviewable commits.
4. After acceptance, follow the venue's instructions for the final tag,
   archival deposit, metadata, and citation record.
5. Report a DOI, publication state, or archive URL only after it is returned by
   the external service and independently verified.

Any APC waiver, institutional agreement, or open-access discount depends on
the journal, article type, author affiliation, and policy in force at the time
of submission. Confirm it with the publisher or institution; this repository
cannot guarantee a zero-cost publication.

## 6. Final release checklist

- [ ] The exact submission revision is recorded.
- [ ] All local validation gates pass on the submission revision.
- [ ] The optional-engine status is archived with the generated data.
- [ ] The manuscript PDF has been rendered and visually inspected page by
      page.
- [ ] Figures and tables agree with the current CSV data.
- [ ] The license, author metadata, citations, and disclosures have been
      checked by the authors.
- [ ] External acceptance, DOI, and publication metadata are reported only
      after confirmation from the relevant service.
