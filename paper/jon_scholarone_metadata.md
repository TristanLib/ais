# ScholarOne Submission Fields

Use this file as the copy-paste source when entering metadata in ScholarOne for
The Journal of Navigation.

## Journal and Article Type

- Journal: The Journal of Navigation
- Article type: Research Article
- Publication route: regular publication unless open-access funding is confirmed

## Title

A Reproducible AIS Trajectory Prediction Benchmark for Navigation Risk-Warning Support

## Short Title

AIS Benchmark for Risk Warning

## Author

- Name: Li Bo
- Affiliation: China Maritime Service Center, China
- Email: li.bo@cmaritime.com.cn
- Corresponding author: Yes
- ORCID: add in ScholarOne if available

## Abstract

Short-horizon AIS trajectory prediction is often framed as a machine-learning problem, but navigation use also requires strong baselines, reproducible splits and downstream warning evidence. This study reports an auditable benchmark built from NOAA historical AIS data on four 2024 dates, containing 186,326 trajectory windows from 7,425 MMSI values. Each sample uses 30 one-minute history points to forecast 15 one-minute positions. Kinematic, Kalman-style, linear and neural baselines are evaluated with Haversine ADE/FDE and local-component RMSE/MAE, followed by CPA/TCPA risk-warning metrics. Kalman-CV gives the best mean ADE in the reported benchmark run: 1,759.7 m on the temporal holdout and 3,109.4 m on the vessel-disjoint holdout. Under the documented baseline configurations, the neural sequence baselines do not outperform the strongest motion baselines. In 2,000 AIS-derived encounter scenarios, Kalman-CV reaches precision 0.963 and recall 0.900. The contribution is a reproducible navigation benchmark, not autonomous collision-avoidance validation.

## Suggested Keywords

- AIS
- maritime navigation
- trajectory prediction
- risk warning

## Funding Statement

This research received no specific grant from any funding agency, commercial or not-for-profit sectors.

## Competing Interests

The author declares no competing interests.

## Data and Code Availability

The code, configuration files, generated figures, compact evidence artefacts and manuscript-generation workflow are available at https://github.com/TristanLib/ais, archived under tag `jon-submission-v1.3`. No separate archival DOI is available for this release at the time of submission. The source data are derived from public NOAA MarineCadastre.gov historical AIS files subject to NOAA data access terms; the repository does not redistribute raw NOAA AIS files or processed NumPy arrays.

## AI-Use Declaration

Generative AI tools were used in May 2026 for assistance with code drafting, language editing, manuscript structuring and consistency checks against repository evidence files. The author verified all numerical results and is responsible for the final content.

## Suggested File Upload Roles

| File | Suggested ScholarOne role |
|---|---|
| `paper/jon_manuscript.docx` | Main manuscript |
| `paper/jon_cover_letter.md` | Cover letter text source |
| `paper/jon_supplementary_materials.zip` | Supplementary material |
| `paper/figures/jon_*.png` | Separate figure files if ScholarOne requests them |
| `paper/jon_manuscript.pdf` | Author review copy only, unless the system requests a PDF |

## Reviewer Suggestions

Leave blank unless suitable reviewers have been checked for conflicts of
interest. Do not suggest collaborators, recent co-authors, supervisors,
students, close colleagues, or anyone with a direct institutional or financial
conflict.
