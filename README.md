# Registration-Tolerant Two-View Phase-Only Holography under Calibration Uncertainty

Companion code, revised manuscripts, technical reports, and result provenance
for a numerical study by Nan-Hong Kuo.

Repository:
<https://github.com/kuonanhong/Registration-Tolerant-Two-View-Phase-Only-Holography-under-Calibration-Uncertainty>

Package version: `2026.09.27-reconstruction`. This is a prepared source
package; the version label does not assert that a GitHub release or an archival
DOI has already been published.

## Read this first

The supplied paper describes a reproducibility archive, but the supplied ZIP
contains LaTeX, generated table fragments, and five figure PDFs. It does not
contain the original Python, target arrays, optimized commands, raw evaluation
records, or loss histories. On 27 September 2026 the target GitHub repository
contained only its title in `README.md`.

This package therefore contains an independent implementation of the stated
model, with explicit target rasterization and random-number conventions.
It does not claim an exact rerun of the unavailable original program. Values
transcribed from the original manuscript are separated from values computed by
the new implementation. A similar-looking plot is not evidence of numerical
reproduction.

All optical results are synthetic simulations. The model uses two separate
time slots, each with its own phase command array. It does not establish
simultaneous two-user display operation, measured eye-box enlargement, an HOE
transfer function, or measured LCoS performance.

## What the study asks

How does optimizing over receiver shifts and a simple uncertain phase-response
model change image shape, addressed-window power, and power spilled into the
inactive channel? The principal mechanism is the tradeoff between registration
tolerance and nominal-image sharpness. A fixed-calibration weighted-target
identity provides an essential low-cost comparator.

The supplied manuscript reports a 16.800% paired shape-error reduction for its
equal-transform-budget five-scenario method, accompanied by 4.189 times the
wrong-window power. It also reports that a simpler collapsed weighted-target
baseline achieves comparable shape error with much less spill. These are
historical reported values, not independent confirmations by this package.
Read the revised paper and technical report for the newly computed results and
the exact limitations of the reconstruction.

## Documents

| Document | PDF | LaTeX source |
|---|---|---|
| Revised English paper | [paper_EN.pdf](output/paper_EN.pdf) | [paper_EN.tex](manuscript/paper_EN.tex) |
| 修訂中文論文 | [paper_ZH.pdf](output/paper_ZH.pdf) | [paper_ZH.tex](manuscript/paper_ZH.tex) |
| English technical report | [technical_report_EN.pdf](output/technical_report_EN.pdf) | [technical_report_EN.tex](report/technical_report_EN.tex) |
| 中文技術報告 | [technical_report_ZH.pdf](output/technical_report_ZH.pdf) | [technical_report_ZH.tex](report/technical_report_ZH.tex) |

The technical reports explain every original figure and table, result
provenance, numerical checks, scientific claims, related work, and journal
submission considerations. Repository instructions are in
[docs/REPRODUCTION.md](docs/REPRODUCTION.md), and evidence categories are in
[docs/PROVENANCE.md](docs/PROVENANCE.md).

## Repository structure

| Path | Purpose |
|---|---|
| `code/model.py` | Explicit targets, Fourier model, objective, gradient, GS, projected Adam, and weighted-target collapse |
| `code/verify.py` | Parseval, inverse, blaze, directional derivative, self-target, and collapse checks |
| `code/verify_outputs.py` | Saved-command forward replay, summary regeneration, and power/command bounds |
| `code/audit_aggregation.py` | Independent raw-to-table statistics, scenario matching, budget and source-hash audit |
| `code/run_study.py` | New simulations and raw/seed-level result exports |
| `code/make_assets.py` | New simulation figures and Tables 1–3 in machine-readable and LaTeX forms |
| `code/reported_tables.py` | Transcription of supplied reported tables, kept apart from simulated records |
| `code/calibration_discrepancy.py` | New post-optimization diagnostic for calibration-induced loss discrepancy |
| `code/verify_math_extension.py` | Independent algebra check for the new diagnostic inequality |
| `code/summarize_sensitivity.py` | Cross-content and panel-size summaries, plus original-versus-new comparison |
| `results_reconstructed/` | Newly computed primary results, inputs, commands, and execution metadata |
| `artifacts_reconstructed/` | Figures and tables generated from the new results |
| `results_reported/` | Historical manuscript values; not recovered raw data |
| `archive/original_submission/` | Supplied LaTeX, original tables, and original five figure PDFs |
| `manuscript/` and `report/` | Revised English and Chinese document sources |
| `output/` | Compiled document PDFs |
| `docs/` | Reproduction, provenance, and release instructions |

Optional panel-size or content studies can be generated in separate directories.
Their names should describe the changed conditions. Never overwrite the
primary results with a sensitivity run.

This prepared package also includes executed `results_panel256/`,
`results_lines/`, and `results_texture/` sensitivity runs. Read each run's own
configuration for its seed count and methods; they are not pooled into the
primary six-seed confidence intervals.

## Quick start

From the repository root, with Python 3.12 and the packages in
`requirements.txt` installed:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/verify.py --out results_check/validation.json
```

Windows PowerShell activation is `.venv\Scripts\Activate.ps1`. A passed check
verifies the tested discrete numerical identities and derivatives. It does not
validate a physical display or certify a globally optimal hologram.

To recompute the new six-seed study into a fresh directory and regenerate its
assets:

```bash
python code/run_study.py --out results_rerun
python code/make_assets.py --results results_rerun --out artifacts_rerun
python code/audit_aggregation.py --results results_rerun --assets artifacts_rerun
```

To export the original reported numbers for comparison:

```bash
python code/reported_tables.py --out results_reported
```

The last command transcribes reported numbers; it does not simulate them.
Full command details and result dependencies are in
[docs/REPRODUCTION.md](docs/REPRODUCTION.md).

To compile all four documents from the archived sources and figures, install
the TeX dependencies described in that guide, then run:

```bash
python scripts/build_documents.py
```

The outputs are written to `output/`. This compiles the archived study; it does
not replace prose claims automatically after a changed simulation.

## Figures and tables

| Asset | Meaning | Generating source |
|---|---|---|
| `angular_channels.pdf` | Absolute angular intensity for one temporal slot | `code/make_assets.py`, using saved commands and `code/model.py` |
| `reconstruction_comparison.pdf` | Local receiver images normalized by each image's own useful power | `code/make_assets.py` |
| `robustness_sweep.pdf` | Shape error, useful power, and spill versus evaluation shift radius | `code/make_assets.py`, using seed-level metrics |
| `collapsed_diagnostic.pdf` | Fixed-calibration weighted-target comparator at a matched transform budget | `code/make_assets.py` |
| `optimization_history.pdf` | Each method's own objective versus updates and model transform pairs | `code/make_assets.py`, using saved histories |
| Table 1 | Prescribed design scenarios; these are inputs | `code/model.py` and `code/make_assets.py` |
| Table 2 | Primary evaluation means and Student-t confidence half-widths across seeds | `code/run_study.py` and `code/make_assets.py` |
| Table 3 | Evaluation-only calibration/registration ablation | `code/run_study.py` and `code/make_assets.py` |
| `calibration_discrepancy.pdf` | New calibration-discrepancy bound and registration-transfer interpretation | `code/calibration_discrepancy.py` |

These script names identify the new implementation. The original figure
PDFs have no accompanying original plotting source in the supplied archive.
The current package does not retroactively attribute them to the new scripts.
The supplied `main.tex` never includes `optimization_history.pdf`; its absence
from that paper is a missing inclusion, not evidence of a failed PDF renderer.
Training-objective values from different design objectives should not be ranked
as though they were a shared test metric.

The calibration-discrepancy diagnostic is a new analysis in this revision,
separate from the supplied primary experiment. It uses a standard weighted-norm
inequality to quantify a model-specific loss discrepancy; it is not a new
distributionally robust guarantee or evidence of measured device calibration.

## Reproducibility and interpretation

- Incident power is one; the illustrative throughput multiplier is 0.86.
  Shape normalization is separate from absolute-power accounting.
- The two command arrays are temporally separate and independently optimized.
- Main comparisons share target power, initializations, and evaluation samples;
  the GS comparator has a different objective and is not a fully controlled
  efficiency comparison.
- Means and confidence intervals use seed-level aggregation. Multiple slots
  and perturbations are not treated as independent manufactured devices.
- Calibration ranges are assumed, not measured. The five training scenarios
  are a coupled design ensemble, not a full independent uncertainty distribution.
- The original fixed-calibration diagnostic and spill-weight sweep were
  exploratory analyses. This history must be retained when discussing novelty.

## Citation, license, and release

Use [CITATION.cff](CITATION.cff), cite the accompanying paper, and record the
exact commit or archived release used for an experiment. No article DOI or
archival DOI is invented in this package.

New software and repository documentation are licensed under MIT; see
[LICENSE](LICENSE), [license.txt](license.txt), and
[LICENSE_SCOPE.md](LICENSE_SCOPE.md). Original supplied articles, figure PDFs,
and third-party materials are outside that software-license scope. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

To publish this prepared package in the existing repository, follow
[docs/GITHUB_RELEASE.md](docs/GITHUB_RELEASE.md). The package itself does not
assert that a remote upload, journal submission, or arXiv submission has occurred.