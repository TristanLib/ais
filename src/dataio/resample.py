"""
AIS data resampling and interpolation utilities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class AISResampler:
    """Resample AIS data to uniform time intervals with interpolation."""

    def __init__(self, config: Dict):
        """Initialize resampler with configuration."""
        self.config = config
        self.processing_config = config['data']['processing']

        self.dt_minutes = self.processing_config['dt_minutes']
        self.interpolation_limit = self.processing_config['interpolation_limit']

    def resample_vessel_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample single vessel trajectory to uniform time intervals.

        Args:
            df: AIS data for single vessel (MMSI), sorted by timestamp

        Returns:
            Resampled DataFrame with uniform time intervals
        """
        if len(df) < 2:
            logger.warning("Insufficient data points for resampling")
            return pd.DataFrame()

        # Ensure timestamp is datetime
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Create uniform time grid
        start_time = df['timestamp'].iloc[0]
        end_time = df['timestamp'].iloc[-1]
        time_grid = pd.date_range(
            start=start_time,
            end=end_time,
            freq=f'{self.dt_minutes}min'
        )

        # Create target DataFrame
        resampled_df = pd.DataFrame({'timestamp': time_grid})
        resampled_df['mmsi'] = df['mmsi'].iloc[0]

        # Interpolate each numeric column
        numeric_cols = ['lat', 'lon', 'sog']
        circular_cols = ['cog']  # Handle circular interpolation separately

        # Linear interpolation for position and SOG
        for col in numeric_cols:
            if col in df.columns:
                resampled_df[col] = self._interpolate_linear(
                    df['timestamp'], df[col], time_grid
                )

        # Circular interpolation for COG
        for col in circular_cols:
            if col in df.columns:
                resampled_df[col] = self._interpolate_circular(
                    df['timestamp'], df[col], time_grid
                )

        # Add derived features
        if 'cog' in resampled_df.columns:
            resampled_df['cog_sin'] = np.sin(np.deg2rad(resampled_df['cog']))
            resampled_df['cog_cos'] = np.cos(np.deg2rad(resampled_df['cog']))

        # Copy categorical columns (forward fill)
        categorical_cols = ['ship_type', 'nav_status', 'length', 'width']
        for col in categorical_cols:
            if col in df.columns:
                resampled_df[col] = df[col].iloc[0]  # Use first value

        # Mark interpolated points
        original_timestamps = set(df['timestamp'])
        resampled_df['is_interpolated'] = ~resampled_df['timestamp'].isin(original_timestamps)

        # Calculate interpolation ratio
        n_interpolated = resampled_df['is_interpolated'].sum()
        resampled_df['interp_ratio'] = n_interpolated / len(resampled_df)

        return resampled_df

    def _interpolate_linear(self, timestamps: pd.Series, values: pd.Series,
                           target_timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Linear interpolation for continuous variables."""
        # Convert timestamps to seconds since start
        t_orig = (timestamps - timestamps.iloc[0]).dt.total_seconds().values
        t_target = (target_timestamps - timestamps.iloc[0]).total_seconds().values

        # Remove NaN values
        try:
            if np.iscomplexobj(values.values):
                # For complex values, check if either real or imaginary part is NaN
                valid_mask = ~(np.isnan(values.values.real) | np.isnan(values.values.imag))
            else:
                valid_mask = ~np.isnan(values.values)
        except:
            # Fallback: assume all values are valid
            valid_mask = np.ones(len(values), dtype=bool)

        if valid_mask.sum() < 2:
            return np.full(len(target_timestamps), np.nan)

        t_valid = t_orig[valid_mask]
        v_valid = values.values[valid_mask]

        # Interpolate
        try:
            f = interp1d(t_valid, v_valid, kind='linear',
                        bounds_error=False, fill_value=np.nan)
            return f(t_target)
        except Exception as e:
            logger.warning(f"Linear interpolation failed: {e}")
            return np.full(len(target_timestamps), np.nan)

    def _interpolate_circular(self, timestamps: pd.Series, angles: pd.Series,
                             target_timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Circular interpolation for angular variables (COG)."""
        # Convert to radians and then to complex numbers
        angles_rad = np.deg2rad(angles)
        complex_angles = np.exp(1j * angles_rad)

        # Interpolate real and imaginary parts separately
        real_interp = self._interpolate_linear(timestamps, pd.Series(complex_angles.real), target_timestamps)
        imag_interp = self._interpolate_linear(timestamps, pd.Series(complex_angles.imag), target_timestamps)

        # Convert back to angles
        complex_interp = real_interp + 1j * imag_interp
        angles_interp = np.angle(complex_interp)

        # Convert back to degrees and ensure [0, 360) range
        angles_deg = np.rad2deg(angles_interp)
        angles_deg = (angles_deg + 360) % 360

        return angles_deg

    def resample_all_vessels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample all vessels in the dataset.

        Args:
            df: Multi-vessel AIS DataFrame

        Returns:
            Resampled DataFrame for all vessels
        """
        logger.info(f"Resampling {df['mmsi'].nunique()} vessels")

        resampled_dfs = []
        failed_vessels = []

        for mmsi, vessel_df in df.groupby('mmsi'):
            try:
                resampled_vessel = self.resample_vessel_trajectory(vessel_df)
                if len(resampled_vessel) > 0:
                    resampled_dfs.append(resampled_vessel)
                else:
                    failed_vessels.append(mmsi)
            except Exception as e:
                logger.warning(f"Failed to resample vessel {mmsi}: {e}")
                failed_vessels.append(mmsi)

        if failed_vessels:
            logger.warning(f"Failed to resample {len(failed_vessels)} vessels")

        if not resampled_dfs:
            logger.error("No vessels successfully resampled")
            return pd.DataFrame()

        # Combine all resampled trajectories
        result_df = pd.concat(resampled_dfs, ignore_index=True)

        logger.info(f"Resampling complete: {len(result_df)} records for {result_df['mmsi'].nunique()} vessels")
        return result_df

    def filter_by_trajectory_length(self, df: pd.DataFrame,
                                   min_length_minutes: int = 120) -> pd.DataFrame:
        """
        Filter vessels by minimum trajectory length.

        Args:
            df: Resampled AIS DataFrame
            min_length_minutes: Minimum trajectory length in minutes

        Returns:
            Filtered DataFrame
        """
        min_points = min_length_minutes // self.dt_minutes

        vessel_lengths = df.groupby('mmsi').size()
        valid_vessels = vessel_lengths[vessel_lengths >= min_points].index

        filtered_df = df[df['mmsi'].isin(valid_vessels)].copy()

        logger.info(f"Filtered by trajectory length: {len(filtered_df)} records "
                   f"for {filtered_df['mmsi'].nunique()} vessels "
                   f"(min {min_length_minutes} minutes)")

        return filtered_df

    def smooth_trajectories(self, df: pd.DataFrame,
                           window_size: int = 5) -> pd.DataFrame:
        """
        Apply smoothing to trajectories to reduce noise.

        Args:
            df: Resampled AIS DataFrame
            window_size: Smoothing window size in time steps

        Returns:
            Smoothed DataFrame
        """
        if window_size <= 1:
            return df

        logger.info(f"Smoothing trajectories with window size {window_size}")

        smoothed_dfs = []

        for mmsi, vessel_df in df.groupby('mmsi'):
            vessel_df = vessel_df.copy().sort_values('timestamp')

            # Smooth position coordinates
            for col in ['lat', 'lon']:
                if col in vessel_df.columns:
                    vessel_df[f'{col}_smooth'] = vessel_df[col].rolling(
                        window=window_size, center=True, min_periods=1
                    ).mean()

            # Smooth SOG (but not COG due to circular nature)
            if 'sog' in vessel_df.columns:
                vessel_df['sog_smooth'] = vessel_df['sog'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()

            smoothed_dfs.append(vessel_df)

        result_df = pd.concat(smoothed_dfs, ignore_index=True)

        # Option to replace original values with smoothed ones
        for col in ['lat', 'lon', 'sog']:
            if f'{col}_smooth' in result_df.columns:
                result_df[col] = result_df[f'{col}_smooth']
                result_df.drop(f'{col}_smooth', axis=1, inplace=True)

        return result_df

    def get_resampling_stats(self, df: pd.DataFrame) -> Dict:
        """Get resampling statistics."""
        stats = {
            'total_vessels': df['mmsi'].nunique(),
            'total_points': len(df),
            'avg_points_per_vessel': len(df) / df['mmsi'].nunique(),
            'interpolation_rate': df['is_interpolated'].mean(),
            'time_coverage': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max(),
                'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            },
            'sampling_interval_minutes': self.dt_minutes
        }

        return stats


if __name__ == "__main__":
    # Example usage
    import yaml

    # Load configuration
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize resampler
    resampler = AISResampler(config)

    # Process example data
    # df_clean = pd.read_parquet("data/interim/cleaned_ais.parquet")
    # df_resampled = resampler.resample_all_vessels(df_clean)
    # stats = resampler.get_resampling_stats(df_resampled)
    # print(stats)
