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

To fully regenerate the evidence package, download the required NOAA AIS files
into `data/raw/`, then run the commands in `REPRODUCIBILITY.md`.
