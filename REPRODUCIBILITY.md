# Reproducibility Guide

This project supports two evidence routes:

- A conservative single-source-date publication pipeline.
- A higher-quality multiday pipeline used for the current
  The Journal of Navigation submission-candidate package.

Run all commands from the repository root.

## Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Conservative Evidence Pipeline

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_final_experiment.sh
```

Smoke test only:

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_final_experiment.sh --max-train-samples 1000 --max-test-samples 100
```

Smoke-test outputs are not manuscript evidence.

## High-Quality / JON Evidence Pipeline

The high-quality pipeline can download the planned NOAA AIS source dates before
building the processed dataset. This is the closest equivalent to a data
installation step:

```bash
PYTHON_BIN=.venv/bin/python \
DOWNLOAD_DATES=true \
bash scripts/run_high_quality_pipeline.sh
```

By default this uses the `planned_dates` in
`configs/experiment_multiday.yaml`. To reproduce the current JON candidate
explicitly, use:

```bash
PYTHON_BIN=.venv/bin/python \
DOWNLOAD_DATES=true \
DATES="2024-01-02 2024-01-09 2024-02-06 2024-03-05" \
bash scripts/run_high_quality_pipeline.sh
```

If the raw files are already present under `data/raw/`, the same pipeline can
be run without re-downloading:

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_high_quality_pipeline.sh
```

Infrastructure-only smoke test:

```bash
PYTHON_BIN=.venv/bin/python \
DOWNLOAD_DATES=true \
MAX_ROWS_PER_FILE=1000000 \
MAX_TRAIN_SAMPLES=512 \
MAX_TEST_SAMPLES=128 \
bash scripts/run_high_quality_pipeline.sh
```

The capped run checks code paths only and must not be cited as paper evidence.

## Regenerate the JON Submission Candidate

```bash
.venv/bin/python scripts/make_jon_submission_pack.py
pandoc --resource-path=paper paper/jon_manuscript.md -o paper/jon_manuscript.docx
pandoc --resource-path=paper paper/jon_manuscript_zh.md -o paper/jon_manuscript_zh.docx
pandoc --resource-path=paper paper/jon_manuscript_zh_interpretation.md -o paper/jon_manuscript_zh_interpretation.docx
```

PDF exports use the bundled document runtime in local Codex sessions:

```bash
/Users/yuebao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/export_markdown_pdf.py \
  --input paper/jon_manuscript.md \
  --output paper/jon_manuscript.pdf \
  --page-size a4 \
  --orientation portrait \
  --serif \
  --margin-inch 1.0
```

Use `--cjk` instead of `--serif` for Chinese PDF exports.

## Validity Boundary

The current evidence supports historical AIS trajectory-prediction benchmarking,
temporal and vessel-disjoint holdout evaluation, and AIS-derived CPA/TCPA
risk-warning metrics. It does not validate live AIS deployment, autonomous
collision avoidance, COLREGs compliance, all-day traffic generalisation, or
general neural architecture superiority.
