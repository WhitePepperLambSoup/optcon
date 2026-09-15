# optcon Release & Publication Submission Guide

This guide provides step-by-step instructions for publishing **optcon** and submitting manuscripts to **JOSS** (Journal of Open Source Software) and/or **CPC** (Computer Physics Communications).

---

## 1. Prerequisites Checklist

Before submitting to any journal or indexing service, run the gates below from the repository root:
- [x] 419 tests collected and passing locally: `python -m pytest tests -q`.
- [x] Core-library coverage is 88% statement coverage when examples and benchmark entry points are excluded by `pyproject.toml`: `python -m pytest tests --cov=optcon --cov-report=term-missing`.
- [x] Type checking: `python -m mypy .` (0 errors across 88 source files in the latest local run).
- [x] Linter: `python -m ruff check .`.
- [x] Reproducibility suite: `python -m optcon.benchmarks.run_all`.
- [x] Optional-engine status: `python -m optcon.benchmarks.engine_status --strict`.
- [x] OSI License: MIT License in `LICENSE`.
- [x] JOSS metadata: `paper.md` and `paper.bib` with complete author affiliations.
- [x] CITATION file: `CITATION.cff` with metadata and BibTeX entry.
- [x] Contribution guidelines: `CONTRIBUTING.md`.

---

## 2. GitHub Repository Initialization

If the repository is not yet public on your GitHub account:

```bash
# 1. Initialize git and commit all work
git init
git add .
git commit -m "feat: initial release candidate v0.1.0 with complete JOSS paper and verification suite"

# 2. Create a public repository on GitHub named 'optcon'
# 3. Add the remote and push
git remote add origin https://github.com/WhitePepperLambSoup/optcon.git
git branch -M main
git push -u origin main
```

---

## 3. JOSS Pre-Submission Gatekeeping Requirements

Before submitting to JOSS, review the [JOSS Submitting Guidelines](https://joss.readthedocs.io/en/latest/submitting.html):
- **Public Development Track Record**: JOSS reviews established research software. Submissions from brand-new, zero-history repositories are routinely rejected at pre-review. Establish a public repository, publish early releases, and maintain ongoing commit activity (JOSS editors typically expect ~6 months of public evolution or a proven development lifecycle).
- **Evidence of Real Research Use**: The software must demonstrate actual research application (e.g., cited in a student thesis, preprinted paper, lab experiment at ANU, or applied research workflow) rather than hypothetical future utility.
- **Generative AI Policy**: JOSS requires full disclosure of AI-assisted drafting, coding, or testing, with the human authors assuming complete responsibility. This is documented in `paper.md`.

---

## 4. JOSS Review & Publication Lifecycle (Diamond Open Access: $0)

### Step 4.1: Submit the Paper
1. Ensure your public GitHub repository is accessible.
2. Navigate to: [https://joss.theoj.org/papers/new](https://joss.theoj.org/papers/new)
3. Enter repository URL: `https://github.com/WhitePepperLambSoup/optcon`
4. Select branch `main` and specify path `paper.md`.
5. Submit for editorial pre-review.

### Step 4.2: Open GitHub Peer Review
1. JOSS opens a transparent issue on `openjournals/joss-reviews`.
2. The editorial bot (`@editorialbot`) validates `paper.md` formatting and renders the draft PDF.
3. Two independent reviewers evaluate code quality, documentation, test suite execution, and research claims.
4. Authors respond to reviewer feedback via GitHub comments and commit fixes directly to `main`.

### Step 4.3: Acceptance & Post-Review Archive Deposit (Zenodo DOI)
*Note on timing*: Permanent archiving on Zenodo/Figshare occurs **after** the paper is accepted:
1. When all reviewer checkboxes are completed, the editor instructs you to tag the final accepted release:
   ```bash
   git tag -a v0.1.0 -m "optcon v0.1.0 accepted JOSS release"
   git push origin v0.1.0
   ```
2. Zenodo automatically mints an archive deposit DOI (e.g., `10.5281/zenodo.XXXXXXX`).
3. In the review thread, issue:
   `@editorialbot deposit <zenodo-doi>`
4. The editor verifies the deposit and assigns the official JOSS CrossRef DOI.

---

## 5. Elsevier CPC / SoftwareX Submission Walkthrough

If submitting a methodology paper to **CPC** (Computer Physics Communications):

### Step 5.1: University APC Waiver via CAUL
- Corresponding author must use their official ANU email: `@anu.edu.au`.
- Under the **CAUL Read & Publish Agreement** with Elsevier, Elsevier Hybrid Open Access journals (such as CPC) provide **100% APC waiver ($0 USD)** automatically during the Rights & Access form step upon acceptance.

### Step 5.2: Manuscript Preparation
- The manuscript text is pre-drafted in `docs/CPC_METHODOLOGY.md`.
- Vector publication figures are located in `docs/figures/`:
  - `fig1_semantic_architecture.pdf`
  - `fig2_mie_adjudication.pdf`
  - `fig3_beam_propagation_anomaly.pdf`
  - `fig4_performance_speedup.pdf`
- Format into Elsevier's `elsarticle` LaTeX template if required.
