# Data Availability

This repository does not redistribute the large NOAA AIS source files or the
processed `.npz` arrays used during local experiments.

## Source Data

The AIS source data are public historical AIS files from NOAA
MarineCadastre.gov:

- https://marinecadastre.gov/ais/
- https://www.coast.noaa.gov/digitalcoast/data/marine-cadastre.html

The current JON submission-candidate evidence package was generated from source
dates recorded in `outputs/audit/multiday_data_manifest.json`:

- 2024-01-02
- 2024-01-09
- 2024-02-06
- 2024-03-05

## Files Not Stored in Git

The following files are intentionally excluded from the GitHub repository:

- Raw AIS CSV/ZIP downloads under `data/raw/`
- Processed NumPy arrays under `data/processed/`
- Large split manifests
- Per-sample error CSV files
- Model checkpoints and intermediate tuning outputs
- Local virtual environments and cache files

These files are reproducible or downloadable, but they are too large for a
clean source repository. A future archival release can place selected large
artifacts in GitHub Releases, Zenodo, OSF, or another data repository.

## Data Installation

The repository includes downloader/preparation code. To download the planned
NOAA AIS dates and run the high-quality/JON evidence pipeline:

```bash
PYTHON_BIN=.venv/bin/python \
DOWNLOAD_DATES=true \
bash scripts/run_high_quality_pipeline.sh
```

For an explicit reproduction of the current candidate:

```bash
PYTHON_BIN=.venv/bin/python \
DOWNLOAD_DATES=true \
DATES="2024-01-02 2024-01-09 2024-02-06 2024-03-05" \
bash scripts/run_high_quality_pipeline.sh
```

The download implementation lives in `scripts/build_multiday_dataset.py`; it
uses the NOAA URL template and `planned_dates` recorded in
`configs/experiment_multiday.yaml`.

## Evidence Included in Git

The repository keeps compact evidence artifacts that support the manuscript
claims:

- Dataset manifests and checksums
- Model metric summaries
- Horizon and scenario-slice summaries
- Statistical summaries
- Risk-warning metrics and scenario table
- Readiness and submission manifests
- Figure-generation scripts and generated manuscript figures

## Reproducibility Note

To fully regenerate the evidence package, use the data installation command
above or place the required NOAA AIS files in `data/raw/`, then run the commands
in `REPRODUCIBILITY.md`.
