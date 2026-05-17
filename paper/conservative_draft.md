# Conservative Manuscript Entry Point

The active conservative manuscript is generated from repository artifacts:

- `paper/conservative_manuscript.md`
- `paper/generated_results_summary.md`
- `outputs/final/publication_readiness_report.json`

Regenerate the manuscript and evidence pack with:

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_final_experiment.sh
```

Publication numbers should be copied only from generated artifacts when `outputs/final/run_manifest.json` records `is_debug_run=false` and `outputs/final/publication_readiness_report.json` records `status=pass`.

The older unsupported claims about a 9.4 m LSTM result, a 23,000x training-methodology improvement, GNN/STT/PINN superiority, 5-fold temporal cross-validation, regional generalization, and collision-avoidance success are excluded from the main manuscript until regenerated and archived by the final pipeline.
