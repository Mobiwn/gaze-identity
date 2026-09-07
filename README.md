# 👁️ Gaze Identity — Task-Disjoint Eye-Gaze Identification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/pytest-7%20passed-brightgreen.svg)](#-testing--quality)
[![Lint](https://img.shields.io/badge/ruff-clean-blueviolet.svg)](#-testing--quality)
[![Manager](https://img.shields.io/badge/managed%20with-uv-black.svg)](https://docs.astral.sh/uv/)

> Can eye-gaze behavior identify a **known participant** on **video tasks the model never trained on**? 🔍
>
> This repo answers that one question — carefully — on a local EasyCog eye-tracking cohort, using a strictly **task-disjoint** protocol with saved predictions, provenance, and uncertainty estimates.

**Author:** Mobin Kheibary · **Supervisor:** Dr. Shiva Kamkar

> ⚠️ **Scope, stated up front:** this is an analysis of one local cohort. It does **not** establish clinical validity, population-level biometric performance, or a causal effect of cognitive load. See [⚠️ Limitations](#️-limitations--what-this-project-does-not-claim) and [`COURSE_REPORT.md`](COURSE_REPORT.md).

---

## ✨ Highlights

- 🧩 **Task-disjoint by construction** — train / validation / test tasks never overlap; the code *rejects* any overlapping split.
- 🖼️ **True picture-level analysis** — uses stored `taskN_pic` picture boundaries directly (task 6 keeps its real 6 pictures; never silently split into 10).
- 🧾 **Fully auditable** — every run saves cohort manifest, SHA-256 source hashes, segment exclusions, per-row predictions, config, seed, environment, and implementation hash.
- 📐 **Small, honest baselines** — nearest-centroid + regularized logistic regression, with variance-filtering and scaling fit on training rows only.
- 📊 **Uncertainty included** — participant-level bootstrap CI + subject-label permutation test, not just point estimates.
- 🌍 **Bilingual docs** — English + Persian ([`README_FA.md`](README_FA.md), [`REPORT_FA.md`](REPORT_FA.md)).

---

## 📊 Saved baseline result

Default config [`configs/baseline.json`](configs/baseline.json) — train on `task1`+`task2`, validate on `task3`, test once on `task7`+`task8`:

| Metric (held-out test) | Value |
|---|---|
| 👥 Subjects retained in all 3 partitions | **42** |
| 🧱 Feature rows (train / val / test) | **813 / 409 / 829** |
| 🏆 Selected model (by validation top-1) | **logistic regression, `C=0.1`** |
| 🎯 Top-1 accuracy | **26.66%** (closed-set chance: 2.38%) |
| 🎯 Top-5 accuracy | **59.35%** |
| 🔢 Mean reciprocal rank | **0.417** |
| 📏 Participant-bootstrap 95% CI (top-1) | **19.37% – 34.53%** |
| 🎲 Subject-label permutation p-value (1,000 perms) | **0.001** |

Regenerate these numbers yourself in ~1 minute (after data setup) — they land in `artifacts/task_disjoint_baseline/`.

---

## 🚀 Quickstart

### 1️⃣ Prerequisites

- Python **3.10+**
- [`uv`](https://docs.astral.sh/uv/) (recommended; plain `pip` works too)
- The eye-tracking `.npz` files (see [📂 Data access](#-data-access) — **not** in this repo, ~1.3 GB)

### 2️⃣ Install

```bash
git clone <your-fork-url> gaze-identity
cd gaze-identity
uv sync --extra dev
```

### 3️⃣ Run the full pipeline

```bash
# Inspect data contract + write auditable cohort manifest
uv run python scripts/inspect_data.py

# Run task-disjoint experiment (predictions, metrics, uncertainty)
uv run python scripts/run_experiment.py

# Verify everything still passes
uv run pytest -q
```

Pretty-print the result:

```bash
uv run python -m json.tool artifacts/task_disjoint_baseline/results.json
```

### 4️⃣ Custom data location / config

```bash
uv run python scripts/inspect_data.py --data-dir /path/to/npz_dir --config configs/baseline.json
uv run python scripts/run_experiment.py --data-dir /path/to/npz_dir --output-dir artifacts/my_split
```

All experimental choices (session policy, quality rules, features, splits, seeds) live in [`configs/baseline.json`](configs/baseline.json). New experiment = new config file. 📝

---

## 📂 Data access

> 🔒 **Raw data is intentionally excluded from version control** (see [`.gitignore`](.gitignore)).

This project analyzes **local EasyCog video `.npz` files only**. The dataset paper is the authoritative source:

> Hu et al. (2026). *The EasyCog Dataset: Towards Easier Cognitive Assessment with Passive Video Watching*. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., 10(1), Article 4. DOI: [10.1145/3789682](https://doi.org/10.1145/3789682).

Expected layout (default: sibling directory, override with `--data-dir`):

```text
parent/
├── gaze-identity/                  # ← this repo
└── asreog_filter_order3_all_data/  # ← local .npz files (NOT committed)
    ├── 002_patient-2024_12_04_17_40_08-video.npz
    └── ...
```

Each video NPZ must contain `subject`, `date`, `type`, `task_et` plus stored `taskN_pic` picture arrays. The inspector validates filenames, embedded fields, gaze shapes, and picture lists before anything enters an experiment — see [`docs/data_contract.md`](docs/data_contract.md).

🩺 The patient spreadsheet (`Patient_Info_dataset.xlsx`) is **not** used for modeling and is also excluded from the repo.

---

## 🗂️ Project structure

```text
gaze-identity/
├── 📜 README.md / README_FA.md      Operational guides (EN / FA)
├── 📜 COURSE_REPORT.md / REPORT_FA.md  Concise deliverable reports (EN / FA)
├── 📜 IMPLEMENTATION.md             Plain-language code walkthrough
├── ⚙️ configs/baseline.json        Reproducible experiment configuration
├── 🧠 src/
│   ├── data_contract.py             NPZ validation, sessions, cohort policy
│   ├── revised_features.py          Picture-level gaze features (no AOI grid)
│   ├── protocol.py                  Disjoint-split validation + partitioning
│   ├── revised_models.py            Nearest-centroid + logistic baselines
│   └── revised_metrics.py           Top-k, MRR, bootstrap, permutation test
├── 🏃 scripts/
│   ├── inspect_data.py              Cohort manifest command
│   └── run_experiment.py            Full experiment orchestration
├── 🧪 tests/                        Unit tests (contract, features, splits, metrics)
├── 📚 docs/                         data_contract.md, methodology.md
├── 🗃️ legacy/                       Archived pre-revision code/reports (traceability only)
├── 📓 notebooks/                    Empty — scratch space (ignored outputs)
└── 🔒 .gitignore / LICENSE         MIT + dataset-aware ignore rules
```

Generated outputs (`artifacts/`) are **git-ignored** — they are reproduced from config, not stored.

---

## 🔬 Method in brief

1. **Cohort** — earliest valid session per subject (deterministic, hashed, logged in `cohort_manifest.json`).
2. **Features** — per stored picture: position stats, velocity/acceleration, fixation/saccade summaries, scanpath length, count-normalized entropy. Missing samples linearly interpolated only if ≥2 valid points. No AOI-grid features until screen coordinates are verified.
3. **Protocol** — pictures from disjoint task sets; model selected on validation only; winner refit on train+val, evaluated **once** on held-out tasks.
4. **Evaluation** — top-1 / top-k accuracy, MRR, participant-bootstrap CI, subject-label permutation p-value.

Details: [`IMPLEMENTATION.md`](IMPLEMENTATION.md) · [`docs/methodology.md`](docs/methodology.md) · [`configs/baseline.json`](configs/baseline.json).

---

## 📁 Outputs (regenerated per run)

`artifacts/task_disjoint_baseline/`:

| File | Contents |
|---|---|
| `cohort_manifest.json` | Selected sessions, SHA-256 hashes, exclusion reasons |
| `segment_exclusions.json` | Picture rows excluded from features (+ why) |
| `predictions.csv` | Every held-out prediction with row provenance |
| `results.json` | Config, splits, validation table, test metrics, CI, p-value, feature names, env, implementation hash |
| `run_metadata.json` | Wall-clock timestamp (kept separate from deterministic results) |

---

## 🧪 Testing & quality

```bash
uv run pytest -q                    # 7 tests: contract, features, protocol, metrics
uv run ruff check src scripts tests # clean ✅ (legacy/ excluded — archived code)
```

- `legacy/` is intentionally excluded from lint (see `pyproject.toml`) — it's a frozen archive, not active code.
- CI tip: tests use tiny synthetic fixtures — **no 1.3 GB dataset needed** for `pytest`.

---

## 🗺️ Documentation map

| File | What it is |
|---|---|
| [`COURSE_REPORT.md`](COURSE_REPORT.md) | 📄 Concise deliverable report (question → cohort → method → results → limits) |
| [`REPORT_FA.md`](REPORT_FA.md) | 📄 Same report in Persian |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | 🔧 Practical code explanation (not the old pipeline) |
| [`docs/data_contract.md`](docs/data_contract.md) | 📏 Exact NPZ expectations + cohort policy |
| [`docs/methodology.md`](docs/methodology.md) | 🔬 Protocol definition |
| [`legacy/`](legacy/README.md) | 🗃️ Superseded exploratory work — do not cite as evidence |

---

## ⚠️ Limitations — what this project does *not* claim

- ❌ No clinical / diagnostic validity.
- ❌ No guarantee for other participants, sessions, devices, or tasks.
- ❌ No causal cognitive-load effect (local keys `task1…task9` ≠ verified load labels; mapping to the paper is unresolved).
- ❌ No screen-AOI or physiological pixel-velocity interpretation (coordinate system unverified).
- ✅ Next steps: reconcile task labels with the source paper, temporal-stability analysis on repeat sessions, predeclared alternative splits, independent cohort replication.

---

## 📚 Citation

If you use this code or protocol, please cite the dataset paper:

```bibtex
@article{Hu2026EasyCog,
  author  = {Hu, ...},
  title   = {The EasyCog Dataset: Towards Easier Cognitive Assessment with Passive Video Watching},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume  = {10},
  number  = {1},
  articleno = {4},
  year    = {2026},
  doi     = {10.1145/3789682}
}
```

---

## 📄 License

MIT — see [`LICENSE`](LICENSE). © 2026 Mobin Kheibary.

## 🙏 Acknowledgements

Supervised by **Dr. Shiva Kamkar**. Built with `numpy`, `scikit-learn`, `pandas`, `scipy`, `xgboost`, `matplotlib`/`seaborn`, `pytest`, `ruff`, and `uv`. Thanks to the EasyCog authors for publishing the dataset.

---

<div align="center">

⭐ If this repo helped your research, please star it — and open an issue for questions! ⭐

🇮🇷 [راهنمای فارسی](README_FA.md) · 📄 [گزارش فارسی](REPORT_FA.md)

</div>
