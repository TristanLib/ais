#!/usr/bin/env python3
"""Compute raw-file checksums and patch the multiday data manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/audit/multiday_data_manifest.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    updated = []
    missing = []
    for row in data.get("raw_files", []):
        path = Path(row["path"])
        if not path.exists():
            row["checksum_status"] = "missing_file"
            missing.append(str(path))
            continue
        row["sha256"] = sha256_file(path)
        row["checksum_status"] = "computed_after_build"
        updated.append(str(path))
    data["raw_checksums_updated_at"] = datetime.now(timezone.utc).isoformat()
    data["raw_checksums_update_status"] = {
        "updated": updated,
        "missing": missing,
    }
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Updated raw checksums in {manifest_path}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
