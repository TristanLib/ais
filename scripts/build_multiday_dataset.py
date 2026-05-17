#!/usr/bin/env python3
"""Build a metadata-rich AIS trajectory dataset for high-quality journal experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except Exception as exc:
            raise RuntimeError(f"Could not parse {path}") from exc


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_data_rows(path: Path) -> int:
    """Count CSV data rows, excluding the header."""
    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, line_count - 1)


def systematic_positions(total_rows: int, sample_rows: int) -> set[int] | None:
    """Return deterministic raw-row positions spread across the full file."""
    if sample_rows <= 0 or total_rows <= sample_rows:
        return None
    return set(np.linspace(0, total_rows - 1, sample_rows, dtype=np.int64).tolist())


def parse_time_blocks(block_text: str | None) -> list[tuple[int, int]]:
    """Parse comma-separated HH:MM-HH:MM UTC time blocks into minute ranges."""
    if not block_text:
        return []
    blocks = []
    for block in block_text.split(","):
        block = block.strip()
        if not block:
            continue
        start_text, end_text = block.split("-", maxsplit=1)
        start_hour, start_minute = [int(part) for part in start_text.split(":")]
        end_hour, end_minute = [int(part) for part in end_text.split(":")]
        start = start_hour * 60 + start_minute
        end = end_hour * 60 + end_minute
        if not (0 <= start < 24 * 60 and 0 < end <= 24 * 60 and start < end):
            raise ValueError(f"Invalid time block: {block}")
        blocks.append((start, end))
    return blocks


def filter_time_blocks(frame: pd.DataFrame, time_blocks: list[tuple[int, int]]) -> pd.DataFrame:
    if not time_blocks or frame.empty:
        return frame
    minute_of_day = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    mask = np.zeros(len(frame), dtype=bool)
    for start, end in time_blocks:
        mask |= (minute_of_day >= start) & (minute_of_day < end)
    return frame[mask].copy()


def parse_date_from_path(path: Path) -> str | None:
    match = re.search(r"(\d{4})[_-](\d{2})[_-](\d{2})", path.name)
    if not match:
        return None
    return "-".join(match.groups())


def planned_url(template: str, date_text: str) -> str:
    year, month, day = [int(part) for part in date_text.split("-")]
    return template.format(year=year, month=month, day=day)


def download_planned_dates(config: dict[str, Any], raw_dir: Path, dates: list[str]) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    template = config["data"]["download_url_template"]
    downloaded = []
    for date_text in dates:
        year, month, day = [int(part) for part in date_text.split("-")]
        zip_path = raw_dir / f"AIS_{year}_{month:02d}_{day:02d}.zip"
        csv_path = raw_dir / f"AIS_{year}_{month:02d}_{day:02d}.csv"
        if csv_path.exists():
            downloaded.append(csv_path)
            continue
        if not zip_path.exists():
            url = planned_url(template, date_text)
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_members:
                raise RuntimeError(f"No CSV in {zip_path}")
            member = csv_members[0]
            with archive.open(member) as source, csv_path.open("wb") as target:
                target.write(source.read())
        downloaded.append(csv_path)
    return downloaded


def normalize_chunk(chunk: pd.DataFrame, column_mapping: dict[str, str], source_file: str) -> pd.DataFrame:
    existing = {old: new for old, new in column_mapping.items() if old in chunk.columns}
    chunk = chunk.rename(columns=existing)
    required = ["timestamp", "mmsi", "lat", "lon", "sog", "cog"]
    missing = [name for name in required if name not in chunk.columns]
    if missing:
        raise ValueError(f"{source_file} missing required columns: {missing}")

    keep = [
        "timestamp",
        "mmsi",
        "lat",
        "lon",
        "sog",
        "cog",
        "heading",
        "ship_type",
        "nav_status",
        "length",
        "width",
    ]
    keep = [col for col in keep if col in chunk.columns]
    chunk = chunk[keep].copy()
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    for col in ["mmsi", "lat", "lon", "sog", "cog", "heading", "ship_type", "length", "width"]:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    chunk["source_file"] = source_file
    return chunk


def filter_chunk(chunk: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    mmsi_min, mmsi_max = filters.get("mmsi_range", [1, 999999999])
    lat_min, lat_max = filters["lat_range"]
    lon_min, lon_max = filters["lon_range"]
    sog_min, sog_max = filters["sog_range"]
    cog_min, cog_max = filters["cog_range"]
    mask = (
        chunk["timestamp"].notna()
        & chunk["mmsi"].between(mmsi_min, mmsi_max)
        & chunk["lat"].between(lat_min, lat_max)
        & chunk["lon"].between(lon_min, lon_max)
        & chunk["sog"].between(sog_min, sog_max)
        & chunk["cog"].between(cog_min, cog_max)
    )
    return chunk[mask].copy()


def load_raw_files(
    paths: list[Path],
    config: dict[str, Any],
    max_rows_per_file: int | None,
    chunksize: int,
    skip_raw_checksums: bool = False,
    sample_rows_per_file: int | None = None,
    sample_time_blocks: list[tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    profiles = []
    for path in paths:
        if not path.exists():
            profiles.append({"path": str(path), "exists": False})
            continue

        source_file = str(path)
        total_rows = 0
        kept_rows = 0
        valid_rows_before_sampling = 0
        mmsi_values: set[int] = set()
        time_min = None
        time_max = None
        valid_time_min = None
        valid_time_max = None
        lat_min = math.inf
        lat_max = -math.inf
        lon_min = math.inf
        lon_max = -math.inf
        file_frames = []
        full_row_count = count_csv_data_rows(path) if sample_rows_per_file is not None else None
        selected_positions = (
            systematic_positions(full_row_count, sample_rows_per_file)
            if full_row_count is not None and sample_rows_per_file is not None
            else None
        )
        selected_positions_array = (
            np.fromiter(selected_positions, dtype=np.int64, count=len(selected_positions))
            if selected_positions is not None
            else None
        )
        raw_offset = 0
        rows_selected_for_processing = 0

        for chunk in pd.read_csv(path, chunksize=chunksize):
            original_chunk_len = len(chunk)
            if selected_positions_array is not None:
                positions = np.arange(raw_offset, raw_offset + original_chunk_len)
                chunk = chunk[np.isin(positions, selected_positions_array)]
            elif sample_rows_per_file is not None and full_row_count is not None and full_row_count <= sample_rows_per_file:
                pass
            raw_offset += original_chunk_len
            if chunk.empty:
                continue
            if max_rows_per_file is not None and total_rows >= max_rows_per_file:
                break
            if max_rows_per_file is not None:
                chunk = chunk.iloc[: max_rows_per_file - total_rows]
            rows_selected_for_processing += len(chunk)
            total_rows += len(chunk)
            norm = normalize_chunk(chunk, config["data"]["column_mapping"], source_file)
            clean = filter_chunk(norm, config["data"]["filters"])
            if clean.empty:
                continue
            clean = clean.drop_duplicates(subset=["mmsi", "timestamp"], keep="first")
            valid_rows_before_sampling += len(clean)
            valid_time_min = clean["timestamp"].min() if valid_time_min is None else min(valid_time_min, clean["timestamp"].min())
            valid_time_max = clean["timestamp"].max() if valid_time_max is None else max(valid_time_max, clean["timestamp"].max())
            clean = filter_time_blocks(clean, sample_time_blocks or [])
            if clean.empty:
                continue
            clean["cog_sin"] = np.sin(np.deg2rad(clean["cog"]))
            clean["cog_cos"] = np.cos(np.deg2rad(clean["cog"]))
            clean["source_date"] = clean["timestamp"].dt.date.astype(str)
            kept_rows += len(clean)
            mmsi_values.update(clean["mmsi"].dropna().astype(int).tolist())
            time_min = clean["timestamp"].min() if time_min is None else min(time_min, clean["timestamp"].min())
            time_max = clean["timestamp"].max() if time_max is None else max(time_max, clean["timestamp"].max())
            lat_min = min(lat_min, float(clean["lat"].min()))
            lat_max = max(lat_max, float(clean["lat"].max()))
            lon_min = min(lon_min, float(clean["lon"].min()))
            lon_max = max(lon_max, float(clean["lon"].max()))
            file_frames.append(clean)

        if file_frames:
            frames.append(pd.concat(file_frames, ignore_index=True))
        profiles.append(
            {
                "path": source_file,
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": None if skip_raw_checksums else sha256_file(path),
                "checksum_status": "skipped" if skip_raw_checksums else "computed",
                "profiled_rows": full_row_count if full_row_count is not None else total_rows,
                "rows_selected_for_processing": rows_selected_for_processing if sample_rows_per_file is not None else total_rows,
                "valid_rows_before_sampling": valid_rows_before_sampling,
                "kept_rows": kept_rows,
                "unique_mmsi_count": len(mmsi_values),
                "valid_timestamp_min": None if valid_time_min is None else valid_time_min.isoformat(),
                "valid_timestamp_max": None if valid_time_max is None else valid_time_max.isoformat(),
                "timestamp_min": None if time_min is None else time_min.isoformat(),
                "timestamp_max": None if time_max is None else time_max.isoformat(),
                "lat_min": None if lat_min is math.inf else lat_min,
                "lat_max": None if lat_max == -math.inf else lat_max,
                "lon_min": None if lon_min is math.inf else lon_min,
                "lon_max": None if lon_max == -math.inf else lon_max,
                "date_from_filename": parse_date_from_path(path),
                "sampling": {
                    "strategy": (
                        "stratified_time_blocks"
                        if sample_time_blocks
                        else ("systematic_raw_row" if sample_rows_per_file is not None else ("first_n_rows" if max_rows_per_file is not None else "none"))
                    ),
                    "sample_rows_per_file": sample_rows_per_file,
                    "sample_time_blocks": sample_time_blocks,
                    "max_rows_per_file": max_rows_per_file,
                    "profile_scope": "selected_rows" if sample_rows_per_file is not None or max_rows_per_file is not None or sample_time_blocks else "full_file",
                },
            }
        )

    if not frames:
        return pd.DataFrame(), profiles
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    return df, profiles


def assign_region(lat: float, lon: float, regions: list[dict[str, Any]]) -> str:
    for region in regions:
        lat_min, lat_max = region["lat_range"]
        lon_min, lon_max = region["lon_range"]
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return region["name"]
    return "other"


def circular_difference_degrees(values: np.ndarray) -> np.ndarray:
    diff = np.diff(values)
    return (diff + 180.0) % 360.0 - 180.0


def resample_vessel(vessel: pd.DataFrame, dt_minutes: int, interpolation_limit: int) -> pd.DataFrame:
    """Resample one vessel to an exact minute grid before slicing."""
    vessel = vessel.sort_values("timestamp").copy()
    vessel["timestamp_minute"] = vessel["timestamp"].dt.floor(f"{dt_minutes}min")
    numeric_cols = ["lat", "lon", "sog", "cog", "cog_sin", "cog_cos"]
    meta_cols = ["mmsi", "source_file", "source_date", "ship_type"]
    agg: dict[str, Any] = {col: "mean" for col in numeric_cols if col in vessel.columns}
    agg.update({col: "first" for col in meta_cols if col in vessel.columns})
    per_minute = vessel.groupby("timestamp_minute", as_index=True).agg(agg).sort_index()
    if len(per_minute) < 2:
        return pd.DataFrame()
    grid = pd.date_range(per_minute.index.min(), per_minute.index.max(), freq=f"{dt_minutes}min")
    resampled = per_minute.reindex(grid)
    observed = resampled["lat"].notna()
    for col in numeric_cols:
        if col in resampled.columns:
            resampled[col] = resampled[col].interpolate(method="time", limit=interpolation_limit, limit_direction="both")
    for col in meta_cols:
        if col in resampled.columns:
            resampled[col] = resampled[col].ffill().bfill()
    resampled["timestamp"] = resampled.index
    resampled["is_interpolated"] = ~observed
    resampled = resampled.dropna(subset=["lat", "lon", "sog", "cog_sin", "cog_cos"]).reset_index(drop=True)
    return resampled


def create_samples(df: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    history = int(config["experiment"]["history_steps"])
    forecast = int(config["experiment"]["forecast_steps"])
    stride = int(config["experiment"].get("stride_steps", 5))
    dt_minutes = int(config["experiment"].get("dt_minutes", 1))
    interpolation_limit = int(config["data"].get("filters", {}).get("max_interpolation_gap_minutes", 3))
    max_interpolation_ratio = float(config["data"].get("filters", {}).get("max_window_interpolation_ratio", 0.25))
    min_density = float(config["data"].get("filters", {}).get("min_observation_density_per_minute", 0.05))
    min_raw_points = int(
        config["data"].get("filters", {}).get(
            "min_raw_points_per_track", math.ceil((history + forecast) / max(1, interpolation_limit + 1))
        )
    )
    regions = config["data"].get("regions", [])
    samples: list[dict[str, Any]] = []

    group_keys = ["mmsi", "source_date"] if "source_date" in df.columns else ["mmsi"]
    for group_key, vessel in df.groupby(group_keys, sort=False):
        mmsi = group_key[0] if isinstance(group_key, tuple) else group_key
        if len(vessel) < max(2, min_raw_points):
            continue
        raw_span_minutes = (vessel["timestamp"].max() - vessel["timestamp"].min()).total_seconds() / 60.0
        if raw_span_minutes < (history + forecast - 1) * dt_minutes:
            continue
        if len(vessel) / max(raw_span_minutes, 1.0) < min_density:
            continue
        vessel = resample_vessel(vessel, dt_minutes, interpolation_limit)
        if len(vessel) < history + forecast:
            continue

        for start in range(0, len(vessel) - history - forecast + 1, stride):
            end = start + history + forecast
            window = vessel.iloc[start:end]
            if float(window["is_interpolated"].mean()) > max_interpolation_ratio:
                continue

            hist = window.iloc[:history]
            fut = window.iloc[history:]
            X = hist[["lat", "lon", "sog", "cog_sin", "cog_cos"]].to_numpy(dtype=np.float64)
            y = fut[["lat", "lon"]].to_numpy(dtype=np.float64)
            cog_turn = np.abs(circular_difference_degrees(hist["cog"].to_numpy(dtype=np.float64)))
            turn_intensity = float(np.nanmean(cog_turn)) if len(cog_turn) else 0.0
            avg_sog = float(hist["sog"].mean())
            lat0 = float(hist["lat"].iloc[-1])
            lon0 = float(hist["lon"].iloc[-1])
            samples.append(
                {
                    "X": X,
                    "y": y,
                    "mmsi": int(mmsi),
                    "start_time": hist["timestamp"].iloc[0].isoformat(),
                    "end_time": fut["timestamp"].iloc[-1].isoformat(),
                    "source_date": str(hist["source_date"].iloc[0]),
                    "source_file": str(hist["source_file"].iloc[0]),
                    "avg_sog": avg_sog,
                    "turn_intensity_deg": turn_intensity,
                    "interpolation_ratio": float(window["is_interpolated"].mean()),
                    "region": assign_region(lat0, lon0, regions),
                    "ship_type": int(hist["ship_type"].iloc[0]) if "ship_type" in hist and not pd.isna(hist["ship_type"].iloc[0]) else -1,
                }
            )
    return samples


def split_temporal(samples: list[dict[str, Any]], ratios: dict[str, float]) -> dict[str, np.ndarray]:
    order = np.array(sorted(range(len(samples)), key=lambda idx: samples[idx]["start_time"]))
    n = len(order)
    train_end = int(n * ratios["train_ratio"])
    val_end = train_end + int(n * ratios["val_ratio"])
    split = np.full(n, "test", dtype=object)
    split[order[:train_end]] = "train"
    split[order[train_end:val_end]] = "val"
    split[order[val_end:]] = "test"
    return {"split": split, "train_end_index": train_end, "val_end_index": val_end}


def split_vessel_disjoint(samples: list[dict[str, Any]], ratios: dict[str, float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vessels = np.array(sorted({sample["mmsi"] for sample in samples}))
    rng.shuffle(vessels)
    n = len(vessels)
    train_end = int(n * ratios["train_ratio"])
    val_end = train_end + int(n * ratios["val_ratio"])
    train = set(vessels[:train_end].tolist())
    val = set(vessels[train_end:val_end].tolist())
    result = []
    for sample in samples:
        if sample["mmsi"] in train:
            result.append("train")
        elif sample["mmsi"] in val:
            result.append("val")
        else:
            result.append("test")
    return np.array(result, dtype=object)


def write_split_manifest(path: Path, samples: list[dict[str, Any]], temporal: np.ndarray, vessel: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sample_index",
            "temporal_split",
            "vessel_split",
            "mmsi",
            "start_time",
            "end_time",
            "source_date",
            "region",
            "avg_sog",
            "turn_intensity_deg",
            "interpolation_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, sample in enumerate(samples):
            writer.writerow(
                {
                    "sample_index": idx,
                    "temporal_split": temporal[idx],
                    "vessel_split": vessel[idx],
                    "mmsi": sample["mmsi"],
                    "start_time": sample["start_time"],
                    "end_time": sample["end_time"],
                    "source_date": sample["source_date"],
                    "region": sample["region"],
                    "avg_sog": sample["avg_sog"],
                    "turn_intensity_deg": sample["turn_intensity_deg"],
                    "interpolation_ratio": sample["interpolation_ratio"],
                }
            )


def save_npz(path: Path, samples: list[dict[str, Any]], temporal_split: np.ndarray, vessel_split: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.stack([sample["X"] for sample in samples])
    y = np.stack([sample["y"] for sample in samples])
    metadata = {
        "mmsi": np.array([sample["mmsi"] for sample in samples], dtype=np.int64),
        "start_time": np.array([sample["start_time"] for sample in samples], dtype=object),
        "end_time": np.array([sample["end_time"] for sample in samples], dtype=object),
        "source_date": np.array([sample["source_date"] for sample in samples], dtype=object),
        "source_file": np.array([sample["source_file"] for sample in samples], dtype=object),
        "region": np.array([sample["region"] for sample in samples], dtype=object),
        "ship_type": np.array([sample["ship_type"] for sample in samples], dtype=np.int64),
        "avg_sog": np.array([sample["avg_sog"] for sample in samples], dtype=np.float64),
        "turn_intensity_deg": np.array([sample["turn_intensity_deg"] for sample in samples], dtype=np.float64),
        "interpolation_ratio": np.array([sample["interpolation_ratio"] for sample in samples], dtype=np.float64),
    }

    def subset(split_name: str) -> tuple[np.ndarray, np.ndarray]:
        idx = np.where(temporal_split == split_name)[0]
        return X[idx], y[idx]

    X_train, y_train = subset("train")
    X_val, y_val = subset("val")
    X_test, y_test = subset("test")
    np.savez_compressed(
        path,
        X=X,
        y=y,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        temporal_split=temporal_split,
        vessel_split=vessel_split,
        **metadata,
    )


def summarize_samples(samples: list[dict[str, Any]], temporal_split: np.ndarray, vessel_split: np.ndarray) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0}
    source_dates = sorted({sample["source_date"] for sample in samples})
    regions = sorted({sample["region"] for sample in samples})
    return {
        "sample_count": len(samples),
        "unique_mmsi_count": len({sample["mmsi"] for sample in samples}),
        "source_dates": source_dates,
        "regions": regions,
        "time_range": {
            "start": min(sample["start_time"] for sample in samples),
            "end": max(sample["end_time"] for sample in samples),
        },
        "temporal_split_counts": {name: int((temporal_split == name).sum()) for name in ["train", "val", "test"]},
        "vessel_split_counts": {name: int((vessel_split == name).sum()) for name in ["train", "val", "test"]},
        "avg_sog_knots": {
            "mean": float(np.mean([sample["avg_sog"] for sample in samples])),
            "min": float(np.min([sample["avg_sog"] for sample in samples])),
            "max": float(np.max([sample["avg_sog"] for sample in samples])),
        },
        "turn_intensity_deg": {
            "mean": float(np.mean([sample["turn_intensity_deg"] for sample in samples])),
            "max": float(np.max([sample["turn_intensity_deg"] for sample in samples])),
        },
        "interpolation_ratio": {
            "mean": float(np.mean([sample["interpolation_ratio"] for sample in samples])),
            "max": float(np.max([sample["interpolation_ratio"] for sample in samples])),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_multiday.yaml")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dates", nargs="*", help="Planned dates to download/use, YYYY-MM-DD.")
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--sample-rows-per-file", type=int, default=None)
    parser.add_argument("--sample-time-blocks", help="Comma-separated UTC blocks, e.g. 00:00-02:00,12:00-14:00.")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--skip-raw-checksums", action="store_true", help="Skip raw-file SHA256 for smoke runs only.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.max_rows_per_file is not None and args.sample_rows_per_file is not None:
        raise ValueError("--max-rows-per-file and --sample-rows-per-file are mutually exclusive.")
    sample_time_blocks = parse_time_blocks(args.sample_time_blocks)
    audit_dir = Path(config["experiment"].get("audit_dir", "outputs/audit"))
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_npz = Path(config["data"]["processed_npz"])
    raw_paths = [Path(path) for path in config["data"].get("raw_files", [])]
    if args.download:
        dates = args.dates or config["data"].get("planned_dates", [])
        raw_paths.extend(download_planned_dates(config, Path("data/raw"), dates))
    raw_paths = list(dict.fromkeys(raw_paths))

    df, raw_profiles = load_raw_files(
        raw_paths,
        config,
        args.max_rows_per_file,
        args.chunksize,
        args.skip_raw_checksums,
        args.sample_rows_per_file,
        sample_time_blocks,
    )
    warnings = []
    if df.empty:
        warnings.append("No raw records were loaded after filtering; no NPZ written.")
        samples: list[dict[str, Any]] = []
        temporal = np.array([], dtype=object)
        vessel = np.array([], dtype=object)
    else:
        samples = create_samples(df, config)
        if not samples:
            warnings.append("No trajectory samples were created; check cadence, horizon, and filters.")
            temporal = np.array([], dtype=object)
            vessel = np.array([], dtype=object)
        else:
            temporal_info = split_temporal(samples, config["splits"]["temporal"])
            temporal = temporal_info["split"]
            vessel = split_vessel_disjoint(samples, config["splits"]["vessel_disjoint"], int(config["experiment"].get("seed", 42)))
            save_npz(output_npz, samples, temporal, vessel)
            write_split_manifest(audit_dir / "multiday_split_manifest.csv", samples, temporal, vessel)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "command_options": {
            "download": args.download,
            "dates": args.dates,
            "max_rows_per_file": args.max_rows_per_file,
            "sample_rows_per_file": args.sample_rows_per_file,
            "sample_time_blocks": sample_time_blocks,
            "sampling_strategy": (
                "stratified_time_blocks"
                if sample_time_blocks
                else ("systematic_raw_row" if args.sample_rows_per_file is not None else ("first_n_rows" if args.max_rows_per_file is not None else "none"))
            ),
            "chunksize": args.chunksize,
            "skip_raw_checksums": args.skip_raw_checksums,
        },
        "processed_npz": str(output_npz),
        "processed_npz_exists": output_npz.exists(),
        "processed_npz_sha256": sha256_file(output_npz) if output_npz.exists() else None,
        "raw_files": raw_profiles,
        "dataset_summary": summarize_samples(samples, temporal, vessel),
        "split_policy": {
            "primary": config["data"].get("split_protocol"),
            "temporal": config["splits"]["temporal"],
            "vessel_disjoint": config["splits"]["vessel_disjoint"],
            "notes": [
                "NPZ includes temporal_split and vessel_split arrays.",
                "X_train/y_train etc. use the temporal split by default.",
            ],
        },
        "scenario_slices": config.get("scenario_slices", {}),
        "warnings": warnings,
    }
    (audit_dir / "multiday_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (audit_dir / "split_policy.json").write_text(json.dumps(manifest["split_policy"], indent=2), encoding="utf-8")
    print(f"Multiday dataset manifest written to {audit_dir / 'multiday_data_manifest.json'}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0 if not warnings else 1


if __name__ == "__main__":
    raise SystemExit(main())
