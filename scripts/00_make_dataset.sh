#!/bin/bash
# Script to download and process AIS data for ship trajectory prediction

set -e  # Exit on any error

echo "=== Ship Trajectory Prediction - Data Processing Pipeline ==="
echo "Starting data processing pipeline..."

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
CONFIG_DIR="$PROJECT_ROOT/configs"

echo "Project root: $PROJECT_ROOT"
echo "Data directory: $DATA_DIR"

# Create data directories
mkdir -p "$DATA_DIR"/{raw,interim,processed,meta}

# Set Python path
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Activate conda environment (if exists)
if command -v conda &> /dev/null; then
    if conda env list | grep -q "ship-prediction"; then
        echo "Activating conda environment: ship-prediction"
        eval "$(conda shell.bash hook)"
        conda activate ship-prediction
    else
        echo "Warning: ship-prediction conda environment not found"
        echo "Please create it using: conda env create -f env/conda.yaml"
    fi
fi

# Step 1: Download sample AIS data (if not exists)
echo "Step 1: Checking for AIS data..."

SAMPLE_DATA_URL="https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/01/AIS_2023_01_01.zip"
RAW_DATA_FILE="$DATA_DIR/raw/sample_ais_data.csv"

if [ ! -f "$RAW_DATA_FILE" ]; then
    echo "Sample AIS data not found. Creating synthetic sample..."

    # Create synthetic AIS data for testing
    python3 << EOF
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create synthetic AIS data for testing
np.random.seed(42)

# Parameters
n_vessels = 50
n_hours = 24
dt_minutes = 1

# Generate vessel data
vessels = []
base_time = datetime(2023, 1, 1, 0, 0, 0)

for mmsi in range(123456789, 123456789 + n_vessels):
    # Random starting position (around New York area)
    start_lat = 40.7 + np.random.normal(0, 0.1)
    start_lon = -74.0 + np.random.normal(0, 0.1)

    # Random course and speed
    course = np.random.uniform(0, 360)
    speed = np.random.uniform(5, 25)  # knots

    # Generate trajectory
    lat, lon = start_lat, start_lon

    for minute in range(n_hours * 60):
        timestamp = base_time + timedelta(minutes=minute)

        # Add some course variation
        course += np.random.normal(0, 2)
        course = course % 360

        # Add some speed variation
        speed += np.random.normal(0, 0.5)
        speed = np.clip(speed, 1, 30)

        # Update position (simplified)
        lat += (speed * np.cos(np.radians(course))) / (60 * 60)  # degrees per minute
        lon += (speed * np.sin(np.radians(course))) / (60 * 60 * np.cos(np.radians(lat)))

        vessels.append({
            'timestamp': timestamp,
            'mmsi': mmsi,
            'lat': lat,
            'lon': lon,
            'sog': speed,
            'cog': course,
            'heading': course + np.random.normal(0, 5),
            'ship_type': np.random.choice([30, 31, 32, 35, 37]),  # Various ship types
            'length': np.random.uniform(50, 300),
            'width': np.random.uniform(10, 40)
        })

# Create DataFrame and save
df = pd.DataFrame(vessels)
os.makedirs(os.path.dirname('$RAW_DATA_FILE'), exist_ok=True)
df.to_csv('$RAW_DATA_FILE', index=False)

print(f"Created synthetic AIS data: {len(df)} records for {df['mmsi'].nunique()} vessels")
print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Saved to: $RAW_DATA_FILE")
EOF

else
    echo "AIS data found at: $RAW_DATA_FILE"
fi

# Step 2: Clean and preprocess data
echo "Step 2: Cleaning AIS data..."
python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')

import yaml
import pandas as pd
from dataio.cleaning import AISDataCleaner, load_ais_csv

# Load configuration
with open('$CONFIG_DIR/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load raw data
print("Loading raw AIS data...")
df_raw = pd.read_csv('$RAW_DATA_FILE')
print(f"Loaded {len(df_raw)} raw records")

# Initialize cleaner
cleaner = AISDataCleaner(config)

# Clean data
print("Cleaning data...")
df_clean = cleaner.clean_ais_data(df_raw)

# Generate quality report
report = cleaner.get_data_quality_report(df_raw, df_clean)
print("Data Quality Report:")
for key, value in report.items():
    print(f"  {key}: {value}")

# Save cleaned data
output_path = '$DATA_DIR/interim/cleaned_ais.parquet'
df_clean.to_parquet(output_path)
print(f"Saved cleaned data to: {output_path}")

# Save quality report
import json
with open('$DATA_DIR/meta/data_quality_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)
EOF

# Step 3: Resample data to uniform intervals
echo "Step 3: Resampling data to uniform intervals..."
python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')

import yaml
import pandas as pd
from dataio.resample import AISResampler

# Load configuration
with open('$CONFIG_DIR/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load cleaned data
print("Loading cleaned data...")
df_clean = pd.read_parquet('$DATA_DIR/interim/cleaned_ais.parquet')

# Initialize resampler
resampler = AISResampler(config)

# Resample all vessels
print("Resampling trajectories...")
df_resampled = resampler.resample_all_vessels(df_clean)

# Filter by trajectory length
print("Filtering by trajectory length...")
df_filtered = resampler.filter_by_trajectory_length(df_resampled, min_length_minutes=120)

# Get statistics
stats = resampler.get_resampling_stats(df_filtered)
print("Resampling Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Save resampled data
output_path = '$DATA_DIR/interim/resampled_ais.parquet'
df_filtered.to_parquet(output_path)
print(f"Saved resampled data to: {output_path}")

# Save statistics
import json
with open('$DATA_DIR/meta/resampling_stats.json', 'w') as f:
    json.dump(stats, f, indent=2, default=str)
EOF

# Step 4: Slice trajectories into training samples
echo "Step 4: Slicing trajectories into training samples..."
python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')

import yaml
import pandas as pd
from dataio.slicing import TrajectorySlicer

# Load configuration
with open('$CONFIG_DIR/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load resampled data
print("Loading resampled data...")
df_resampled = pd.read_parquet('$DATA_DIR/interim/resampled_ais.parquet')

# Initialize slicer
slicer = TrajectorySlicer(config)

# Slice trajectories
print("Slicing trajectories...")
samples = slicer.slice_all_trajectories(df_resampled)

# Split into train/val/test
print("Splitting into train/val/test sets...")
train_samples, val_samples, test_samples = slicer.split_samples(samples)

# Get statistics
stats = slicer.get_slicing_stats(samples)
print("Slicing Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Save samples
slicer.save_samples(train_samples, '$DATA_DIR/processed/train_samples.pkl')
slicer.save_samples(val_samples, '$DATA_DIR/processed/val_samples.pkl')
slicer.save_samples(test_samples, '$DATA_DIR/processed/test_samples.pkl')

print(f"Saved {len(train_samples)} train samples")
print(f"Saved {len(val_samples)} validation samples")
print(f"Saved {len(test_samples)} test samples")

# Save statistics
import json
with open('$DATA_DIR/meta/slicing_stats.json', 'w') as f:
    json.dump(stats, f, indent=2, default=str)
EOF

# Step 5: Create data dictionary
echo "Step 5: Creating data dictionary..."
cat > "$DATA_DIR/meta/data_dictionary.yaml" << 'EOF'
# Data Dictionary for Ship Trajectory Prediction Dataset

dataset_info:
  name: "Ship Trajectory Prediction Dataset"
  description: "Processed AIS data for ship trajectory prediction and collision avoidance"
  version: "1.0"
  creation_date: "2024-01-01"

fields:
  timestamp:
    description: "UTC timestamp of AIS message"
    type: "datetime"
    format: "YYYY-MM-DD HH:MM:SS"

  mmsi:
    description: "Maritime Mobile Service Identity (unique vessel identifier)"
    type: "integer"
    range: [100000000, 999999999]

  lat:
    description: "Latitude in decimal degrees"
    type: "float"
    range: [-90.0, 90.0]
    unit: "degrees"

  lon:
    description: "Longitude in decimal degrees"
    type: "float"
    range: [-180.0, 180.0]
    unit: "degrees"

  sog:
    description: "Speed over ground"
    type: "float"
    range: [0.0, 50.0]
    unit: "knots"

  cog:
    description: "Course over ground"
    type: "float"
    range: [0.0, 360.0]
    unit: "degrees"

  cog_sin:
    description: "Sine of course over ground (for circular encoding)"
    type: "float"
    range: [-1.0, 1.0]

  cog_cos:
    description: "Cosine of course over ground (for circular encoding)"
    type: "float"
    range: [-1.0, 1.0]

  ship_type:
    description: "AIS ship type code"
    type: "integer"
    values:
      30: "Fishing"
      31: "Towing"
      32: "Towing (large)"
      33: "Dredging"
      34: "Diving"
      35: "Military"
      36: "Sailing"
      37: "Pleasure craft"

  is_interpolated:
    description: "Whether this record was interpolated"
    type: "boolean"

  interp_ratio:
    description: "Ratio of interpolated points in trajectory segment"
    type: "float"
    range: [0.0, 1.0]

processing_parameters:
  dt_minutes: 1
  history_minutes: 60
  forecast_minutes: 30
  stride_minutes: 10
  coordinate_system: "relative_meters"

splits:
  train: 0.7
  validation: 0.15
  test: 0.15
  split_method: "by_mmsi"
EOF

echo "=== Data Processing Complete ==="
echo "Summary:"
echo "- Raw data: $DATA_DIR/raw/"
echo "- Processed data: $DATA_DIR/processed/"
echo "- Metadata: $DATA_DIR/meta/"
echo ""
echo "Next steps:"
echo "1. Run baseline training: bash scripts/10_train_baselines.sh"
echo "2. Run transformer training: bash scripts/20_train_transformer.sh"
echo "3. Evaluate models: bash scripts/30_eval_all.sh"
