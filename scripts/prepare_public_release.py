#!/usr/bin/env python3
"""Create a clean public-release copy of the project.

The local research workspace contains raw AIS files, processed arrays,
checkpoints, large per-sample outputs, caches, and legacy exploratory artifacts.
This script copies only the source code, documentation, manuscript artifacts,
figures, and compact evidence files suitable for a GitHub repository.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
from pathlib import Path


ROOT_FILES = [
    ".gitattributes",
    ".gitignore",
    "AGENTS.md",
    "CITATION.cff",
    "DATA_AVAILABILITY.md",
    "HIGH_QUALITY_JOURNAL_ROADMAP.md",
    "JOURNAL_OF_NAVIGATION_SUBMISSION_ROADMAP.md",
    "LICENSE",
    "PUBLICATION_CURRENT_STATUS.md",
    "PUBLICATION_IMPLEMENTATION_PLAN.md",
    "PUBLIC_RELEASE_MANIFEST.md",
    "README.md",
    "REPRODUCIBILITY.md",
    "TARGET_JOURNAL_SELECTION.md",
    "requirements.txt",
]

TREE_PATTERNS = [
    "configs/**",
    "data/README.md",
    "data/meta/*.json",
    "paper/conservative_*.md",
    "paper/generated_results_summary.md",
    "paper/figures/jon_*.png",
    "paper/jon_*.md",
    "paper/jon_*.pdf",
    "paper/jon_*.docx",
    "paper/references.bib",
    "paper/submission_manuscript.md",
    "paper/submission_manuscript.pdf",
    "paper/submission_manuscript_zh.md",
    "paper/submission_manuscript_zh.pdf",
    "paper/tables/*.md",
    "scripts/*.py",
    "scripts/*.sh",
    "src/**",
    "outputs/audit/data_manifest.json",
    "outputs/audit/environment.json",
    "outputs/audit/feature_schema.json",
    "outputs/audit/multiday_data_manifest.json",
    "outputs/audit/run_command.txt",
    "outputs/audit/split_policy.json",
    "outputs/final/data_quality_report.json",
    "outputs/final/error_summary_by_group.csv",
    "outputs/final/error_summary_by_horizon.csv",
    "outputs/final/model_metrics.csv",
    "outputs/final/publication_readiness_report.json",
    "outputs/final/reproducibility_check.json",
    "outputs/final/run_manifest.json",
    "outputs/final/statistical_tests.json",
    "outputs/final/figures/*.png",
    "outputs/final/figures/*.json",
    "outputs/final/tables/*",
    "outputs/final_multiday/error_summary_by_group.csv",
    "outputs/final_multiday/error_summary_by_horizon.csv",
    "outputs/final_multiday/generalization_metrics.csv",
    "outputs/final_multiday/model_metrics.csv",
    "outputs/final_multiday/neural_tuning_protocol.json",
    "outputs/final_multiday/neural_tuning_results.csv",
    "outputs/final_multiday/run_manifest.json",
    "outputs/final_multiday/statistical_tests.json",
    "outputs/final_risk/risk_metrics.json",
    "outputs/final_risk/risk_scenarios.csv",
    "outputs/final_risk/figures/*.png",
    "outputs/final_submission/*.json",
    "outputs/latest_predictions/*.csv",
    "outputs/latest_predictions/*.json",
]

EXCLUDE_PATTERNS = [
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/*.pyc",
    "data/raw/**",
    "data/interim/**",
    "data/processed/**",
    "outputs/**/checkpoints/**",
    "outputs/**/training_logs/**",
    "outputs/**/tuning/**",
    "outputs/**/per_sample_errors.csv",
    "outputs/audit/*split_manifest.csv",
    "outputs/models/**",
    ".venv/**",
    "ship_env/**",
]

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def included(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    if rel in ROOT_FILES:
        return True
    if matches(rel, EXCLUDE_PATTERNS):
        return False
    return matches(rel, TREE_PATTERNS)


def copy_file(src: Path, dst_root: Path, root: Path) -> None:
    rel = src.relative_to(root)
    if src.stat().st_size > MAX_FILE_SIZE_BYTES:
        raise RuntimeError(f"Refusing to copy large file over 10 MB: {rel}")
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True, help="Destination directory for the clean public copy.")
    parser.add_argument("--force", action="store_true", help="Remove destination first if it exists.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dest = Path(args.dest).resolve()
    if dest.exists():
        if not args.force:
            raise SystemExit(f"Destination exists: {dest}. Use --force to replace it.")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    copied = 0
    for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
        if included(file_path, root):
            copy_file(file_path, dest, root)
            copied += 1

    print(f"Copied {copied} public-release files to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
