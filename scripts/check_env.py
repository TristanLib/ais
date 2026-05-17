#!/usr/bin/env python3
"""Check and archive the Python environment for publication experiments."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


REQUIRED_MODULES = [
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "matplotlib",
    "yaml",
    "torch",
]


def module_report(module_name: str) -> dict:
    report = {
        "module": module_name,
        "available": False,
        "version": None,
        "error": None,
    }
    try:
        importlib.import_module(module_name)
        report["available"] = True
        package_name = "PyYAML" if module_name == "yaml" else module_name
        if module_name == "sklearn":
            package_name = "scikit-learn"
        try:
            report["version"] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            report["version"] = "unknown"
    except Exception as exc:  # pragma: no cover - deliberately captures env issues
        report["error"] = str(exc)
    return report


def git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def build_report() -> dict:
    modules = [module_report(name) for name in REQUIRED_MODULES]
    missing = [item["module"] for item in modules if not item["available"]]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info),
            "platform": platform.platform(),
        },
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["branch", "--show-current"]),
            "status_short": git_value(["status", "--short"]),
        },
        "modules": modules,
        "missing_required_modules": missing,
        "ok": not missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="outputs/audit/environment.json",
        help="Where to write the environment report.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write the report but return success even when modules are missing.",
    )
    args = parser.parse_args()

    report = build_report()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if report["ok"]:
        print(f"Environment check passed. Report written to {output_path}")
        return 0

    missing = ", ".join(report["missing_required_modules"])
    print(f"Environment check found missing modules: {missing}")
    print(f"Report written to {output_path}")
    return 0 if args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
