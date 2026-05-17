#!/usr/bin/env python3
"""Create auditable dataset manifests for the conservative publication run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except Exception as exc:
            raise RuntimeError(
                f"Could not parse {path}. Install PyYAML or keep the config JSON-compatible."
            ) from exc


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, include_hash: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": None,
        "sha256": None,
    }
    if path.exists():
        record["size_bytes"] = path.stat().st_size
        if include_hash:
            record["sha256"] = sha256_file(path)
    return record


def profile_raw_csv(path: Path) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "profiled": False,
        "record_count": None,
        "unique_mmsi_count": None,
        "timestamp_min": None,
        "timestamp_max": None,
        "columns": [],
    }
    if not path.exists() or path.suffix.lower() != ".csv":
        return profile

    mmsi_values = set()
    timestamp_min = None
    timestamp_max = None
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        profile["columns"] = reader.fieldnames or []
        mmsi_field = "MMSI" if "MMSI" in profile["columns"] else "mmsi" if "mmsi" in profile["columns"] else None
        time_field = (
            "BaseDateTime"
            if "BaseDateTime" in profile["columns"]
            else "timestamp"
            if "timestamp" in profile["columns"]
            else None
        )
        count = 0
        for row in reader:
            count += 1
            if mmsi_field and row.get(mmsi_field):
                mmsi_values.add(row[mmsi_field])
            if time_field and row.get(time_field):
                timestamp = row[time_field]
                timestamp_min = timestamp if timestamp_min is None or timestamp < timestamp_min else timestamp_min
                timestamp_max = timestamp if timestamp_max is None or timestamp > timestamp_max else timestamp_max

    profile.update(
        {
            "profiled": True,
            "record_count": count,
            "unique_mmsi_count": len(mmsi_values) if mmsi_values else None,
            "timestamp_min": timestamp_min,
            "timestamp_max": timestamp_max,
        }
    )
    return profile


def array_record(name: str, array: np.ndarray) -> dict[str, Any]:
    numeric = np.issubdtype(array.dtype, np.number)
    record: dict[str, Any] = {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "numeric": bool(numeric),
    }
    if numeric and array.size:
        finite = np.isfinite(array)
        flat = array.reshape(-1, array.shape[-1]) if array.ndim >= 2 else array.reshape(-1, 1)
        record.update(
            {
                "min": float(np.nanmin(array)),
                "max": float(np.nanmax(array)),
                "mean": float(np.nanmean(array)),
                "finite_ratio": float(finite.mean()),
                "col_min": [float(x) for x in np.nanmin(flat, axis=0)],
                "col_max": [float(x) for x in np.nanmax(flat, axis=0)],
            }
        )
    return record


def infer_position_units(arrays: dict[str, np.ndarray], keys: dict[str, str]) -> dict[str, Any]:
    """Infer whether the first two position dimensions look like WGS84 lat/lon."""
    candidates = [
        keys.get("X_train"),
        keys.get("X_val"),
        keys.get("X_test"),
        keys.get("y_train"),
        keys.get("y_val"),
        keys.get("y_test"),
    ]
    lat_values = []
    lon_values = []
    for key in candidates:
        if not key or key not in arrays:
            continue
        array = arrays[key]
        if not np.issubdtype(array.dtype, np.number) or array.ndim < 2 or array.shape[-1] < 2:
            continue
        flat = array.reshape(-1, array.shape[-1])
        lat_values.append(flat[:, 0])
        lon_values.append(flat[:, 1])

    if not lat_values or not lon_values:
        return {"inferred": "unknown", "reason": "no numeric position columns found"}

    lat = np.concatenate(lat_values)
    lon = np.concatenate(lon_values)
    finite = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(finite):
        return {"inferred": "unknown", "reason": "position columns contain no finite values"}

    lat = lat[finite]
    lon = lon[finite]
    lat_min = float(np.nanmin(lat))
    lat_max = float(np.nanmax(lat))
    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))
    in_wgs84_bounds = -90.0 <= lat_min <= lat_max <= 90.0 and -180.0 <= lon_min <= lon_max <= 180.0
    return {
        "inferred": "degrees_latlon_wgs84" if in_wgs84_bounds else "unknown",
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "finite_ratio": float(finite.mean()),
    }


def write_split_manifest(path: Path, split_counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "sample_index"])
        writer.writeheader()
        for split, count in split_counts.items():
            for idx in range(count):
                writer.writerow({"split": split, "sample_index": idx})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_conservative.yaml")
    parser.add_argument("--hash", action="store_true", help="Compute SHA256 hashes.")
    parser.add_argument("--profile-raw", action="store_true", help="Count raw CSV records and unique MMSI values.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    audit_dir = Path(config["experiment"].get("audit_dir", "outputs/audit"))
    output_dir = Path(config["experiment"].get("output_dir", "outputs/final"))
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = config["data"]
    processed_path = Path(data_config["processed_npz"])
    raw_paths = [Path(item) for item in data_config.get("raw_files", [])]

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "experiment": config["experiment"],
        "data_policy": {
            "split_protocol": data_config.get("split_protocol"),
            "coordinate_units": data_config.get("coordinate_units"),
            "metric_distance": data_config.get("metric_distance"),
            "horizon_description": data_config.get("horizon_description"),
            "target_extension_allowed_for_main_results": data_config.get(
                "target_extension_allowed_for_main_results"
            ),
        },
        "files": {
            "processed_npz": file_record(processed_path, args.hash),
            "raw_files": [],
        },
        "arrays": {},
        "split_counts": {},
        "warnings": [],
    }

    coordinate_units = str(data_config.get("coordinate_units", "")).lower()
    if "unknown" in coordinate_units or "to_be_verified" in coordinate_units:
        manifest["warnings"].append(
            "Coordinate units are not verified. Do not cite ADE/FDE/RMSE as meters until this is resolved."
        )

    for path in raw_paths:
        raw_record = file_record(path, args.hash)
        if args.profile_raw:
            raw_record["profile"] = profile_raw_csv(path)
        manifest["files"]["raw_files"].append(raw_record)

    if not processed_path.exists():
        manifest["warnings"].append(f"Processed NPZ missing: {processed_path}")
    else:
        loaded_arrays: dict[str, np.ndarray] = {}
        with np.load(processed_path, allow_pickle=False) as data:
            for key in data.files:
                loaded_arrays[key] = data[key]
                manifest["arrays"][key] = array_record(key, loaded_arrays[key])

        keys = data_config["array_keys"]
        inferred_units = infer_position_units(loaded_arrays, keys)
        manifest["data_policy"]["inferred_position_units"] = inferred_units
        if inferred_units.get("inferred") != data_config.get("coordinate_units"):
            manifest["warnings"].append(
                "Configured coordinate_units does not match inferred_position_units."
            )

        split_counts = {
            "train": manifest["arrays"][keys["X_train"]]["shape"][0],
            "val": manifest["arrays"][keys["X_val"]]["shape"][0],
            "test": manifest["arrays"][keys["X_test"]]["shape"][0],
        }
        manifest["split_counts"] = split_counts

        forecast_steps = manifest["arrays"][keys["y_test"]]["shape"][1]
        expected_forecast = config["experiment"]["forecast_steps"]
        if forecast_steps != expected_forecast:
            manifest["warnings"].append(
                f"Configured forecast_steps={expected_forecast}, but y_test has {forecast_steps}."
            )

        split_manifest_path = audit_dir / "split_manifest.csv"
        write_split_manifest(split_manifest_path, split_counts)
        manifest["split_manifest"] = str(split_manifest_path)

    feature_schema = {
        "created_at": manifest["created_at"],
        "input_features": [
            {"index": 0, "name": "x_or_lat", "unit": "see_data_manifest"},
            {"index": 1, "name": "y_or_lon", "unit": "see_data_manifest"},
            {"index": 2, "name": "sog", "unit": "knots_or_normalized"},
            {"index": 3, "name": "cog_sin", "unit": "unitless"},
            {"index": 4, "name": "cog_cos", "unit": "unitless"},
        ],
        "target": {
            "dimensions": ["x_or_lat", "y_or_lon"],
            "unit": data_config.get("coordinate_units"),
            "metric_distance": data_config.get("metric_distance"),
        },
        "notes": [
            "The current NPZ does not contain MMSI or timestamp metadata.",
            "Do not claim vessel-disjoint or temporal cross-validation from this NPZ alone.",
        ],
    }

    (audit_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (audit_dir / "feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2), encoding="utf-8"
    )
    data_quality_report = {
        "created_at": manifest["created_at"],
        "source_manifest": str(audit_dir / "data_manifest.json"),
        "processed_npz_exists": manifest["files"]["processed_npz"]["exists"],
        "raw_file_profiles": [
            {
                "path": raw_file.get("path"),
                "record_count": raw_file.get("profile", {}).get("record_count"),
                "unique_mmsi_count": raw_file.get("profile", {}).get("unique_mmsi_count"),
                "timestamp_min": raw_file.get("profile", {}).get("timestamp_min"),
                "timestamp_max": raw_file.get("profile", {}).get("timestamp_max"),
            }
            for raw_file in manifest["files"]["raw_files"]
        ],
        "split_counts": manifest["split_counts"],
        "coordinate_units": manifest["data_policy"].get("coordinate_units"),
        "metric_distance": manifest["data_policy"].get("metric_distance"),
        "inferred_position_units": manifest["data_policy"].get("inferred_position_units"),
        "warnings": manifest["warnings"],
        "metadata_limitations": [
            "The processed NPZ does not include MMSI, timestamp, vessel type, or region fields.",
            "Group-level analysis is limited to all-sample summaries unless a richer manifest is regenerated.",
        ],
    }
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(data_quality_report, indent=2), encoding="utf-8"
    )

    print(f"Dataset audit written to {audit_dir / 'data_manifest.json'}")
    if manifest["warnings"]:
        print("Warnings:")
        for warning in manifest["warnings"]:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
