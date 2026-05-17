"""
AIS data cleaning and preprocessing utilities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AISDataCleaner:
    """Clean and preprocess raw AIS data."""

    def __init__(self, config: Dict):
        """Initialize cleaner with configuration."""
        self.config = config
        self.processing_config = config['data']['processing']

        # Validation ranges
        self.lat_range = self.processing_config['lat_range']
        self.lon_range = self.processing_config['lon_range']
        self.sog_range = self.processing_config['sog_range']
        self.cog_range = self.processing_config['cog_range']

    def clean_ais_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw AIS data.

        Args:
            df: Raw AIS DataFrame with columns [timestamp, mmsi, lat, lon, sog, cog, ...]

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning AIS data: {len(df)} records")
        initial_count = len(df)

        # 1. Remove invalid coordinates
        df = self._filter_coordinates(df)
        logger.info(f"After coordinate filtering: {len(df)} records ({len(df)/initial_count:.3f} retained)")

        # 2. Remove invalid SOG/COG
        df = self._filter_navigation_data(df)
        logger.info(f"After navigation filtering: {len(df)} records ({len(df)/initial_count:.3f} retained)")

        # 3. Remove duplicates
        df = self._remove_duplicates(df)
        logger.info(f"After deduplication: {len(df)} records ({len(df)/initial_count:.3f} retained)")

        # 4. Sort by MMSI and timestamp
        df = df.sort_values(['mmsi', 'timestamp']).reset_index(drop=True)

        # 5. Add derived features
        df = self._add_derived_features(df)

        logger.info(f"Cleaning complete: {len(df)} records retained ({len(df)/initial_count:.3f})")
        return df

    def _filter_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter invalid coordinates."""
        lat_min, lat_max = self.lat_range
        lon_min, lon_max = self.lon_range

        valid_coords = (
            (df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
            (df['lon'] >= lon_min) & (df['lon'] <= lon_max) &
            ~df['lat'].isna() & ~df['lon'].isna()
        )

        return df[valid_coords].copy()

    def _filter_navigation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter invalid SOG and COG values."""
        sog_min, sog_max = self.sog_range
        cog_min, cog_max = self.cog_range

        valid_nav = (
            (df['sog'] >= sog_min) & (df['sog'] <= sog_max) &
            (df['cog'] >= cog_min) & (df['cog'] <= cog_max) &
            ~df['sog'].isna() & ~df['cog'].isna()
        )

        return df[valid_nav].copy()

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records."""
        # Remove exact duplicates
        df = df.drop_duplicates()

        # Remove duplicates by MMSI and timestamp (keep first)
        df = df.drop_duplicates(subset=['mmsi', 'timestamp'], keep='first')

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for model input."""
        # Convert COG to sin/cos to handle circular nature
        df['cog_sin'] = np.sin(np.deg2rad(df['cog']))
        df['cog_cos'] = np.cos(np.deg2rad(df['cog']))

        # Calculate heading (if available, otherwise use COG)
        if 'heading' in df.columns:
            df['heading_sin'] = np.sin(np.deg2rad(df['heading']))
            df['heading_cos'] = np.cos(np.deg2rad(df['heading']))
        else:
            df['heading_sin'] = df['cog_sin']
            df['heading_cos'] = df['cog_cos']

        # Add time features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        return df

    def filter_by_ship_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by ship type if specified in config."""
        ship_config = self.config['data'].get('ship_types', {})

        if 'include' in ship_config:
            include_types = ship_config['include']
            df = df[df['ship_type'].isin(include_types)]
            logger.info(f"Filtered to include ship types: {include_types}")

        if 'exclude' in ship_config:
            exclude_types = ship_config['exclude']
            df = df[~df['ship_type'].isin(exclude_types)]
            logger.info(f"Excluded ship types: {exclude_types}")

        return df

    def get_data_quality_report(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> Dict:
        """Generate data quality report."""
        report = {
            'raw_records': len(df_raw),
            'clean_records': len(df_clean),
            'retention_rate': len(df_clean) / len(df_raw) if len(df_raw) > 0 else 0,
            'unique_vessels': df_clean['mmsi'].nunique(),
            'time_range': {
                'start': df_clean['timestamp'].min(),
                'end': df_clean['timestamp'].max(),
                'duration_hours': (df_clean['timestamp'].max() - df_clean['timestamp'].min()).total_seconds() / 3600
            },
            'geographic_bounds': {
                'lat_min': df_clean['lat'].min(),
                'lat_max': df_clean['lat'].max(),
                'lon_min': df_clean['lon'].min(),
                'lon_max': df_clean['lon'].max()
            },
            'navigation_stats': {
                'sog_mean': df_clean['sog'].mean(),
                'sog_std': df_clean['sog'].std(),
                'sog_range': [df_clean['sog'].min(), df_clean['sog'].max()]
            }
        }

        return report


def load_ais_csv(file_path: Path) -> pd.DataFrame:
    """Load AIS data from CSV file with standard column mapping."""
    try:
        df = pd.read_csv(file_path)

        # Standard column mapping (adjust based on your data source)
        column_mapping = {
            'BaseDateTime': 'timestamp',
            'MMSI': 'mmsi',
            'LAT': 'lat',
            'LON': 'lon',
            'SOG': 'sog',
            'COG': 'cog',
            'Heading': 'heading',
            'VesselType': 'ship_type',
            'Status': 'nav_status',
            'Length': 'length',
            'Width': 'width'
        }

        # Rename columns if they exist
        existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_cols)

        # Ensure required columns exist
        required_cols = ['timestamp', 'mmsi', 'lat', 'lon', 'sog', 'cog']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df

    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    import yaml

    # Load configuration
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize cleaner
    cleaner = AISDataCleaner(config)

    # Process example file
    # df_raw = load_ais_csv("data/raw/sample.csv")
    # df_clean = cleaner.clean_ais_data(df_raw)
    # report = cleaner.get_data_quality_report(df_raw, df_clean)
    # print(report)
