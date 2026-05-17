"""
AIS trajectory slicing for model training and evaluation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class TrajectorySlicerConfig:
    """Configuration for trajectory slicing."""

    def __init__(self, config: Dict):
        self.processing_config = config['data']['processing']
        self.history_minutes = self.processing_config['history_minutes']
        self.forecast_minutes = self.processing_config['forecast_minutes']
        self.stride_minutes = self.processing_config['stride_minutes']
        self.dt_minutes = self.processing_config['dt_minutes']

        # Convert to time steps
        self.history_steps = self.history_minutes // self.dt_minutes
        self.forecast_steps = self.forecast_minutes // self.dt_minutes
        self.stride_steps = self.stride_minutes // self.dt_minutes

        # Coordinate system settings
        self.use_relative_coords = self.processing_config['use_relative_coords']

        # Neighbor settings
        self.neighbor_radius_nmi = self.processing_config['neighbor_radius_nmi']
        self.max_neighbors = self.processing_config['max_neighbors']

        # Data split settings
        self.train_ratio = self.processing_config['train_ratio']
        self.val_ratio = self.processing_config['val_ratio']
        self.test_ratio = self.processing_config['test_ratio']
        self.split_by_mmsi = self.processing_config['split_by_mmsi']


class TrajectorySlicer:
    """Slice vessel trajectories into training samples."""

    def __init__(self, config: Dict):
        """Initialize slicer with configuration."""
        self.config = TrajectorySlicerConfig(config)

    def slice_vessel_trajectory(self, df: pd.DataFrame) -> List[Dict]:
        """
        Slice single vessel trajectory into samples.

        Args:
            df: Resampled trajectory for single vessel, sorted by timestamp

        Returns:
            List of trajectory samples
        """
        if len(df) < self.config.history_steps + self.config.forecast_steps:
            logger.warning(f"Trajectory too short: {len(df)} points")
            return []

        samples = []

        # Sliding window approach
        for i in range(0, len(df) - self.config.history_steps - self.config.forecast_steps + 1,
                      self.config.stride_steps):

            # Extract history and forecast windows
            history_start = i
            history_end = i + self.config.history_steps
            forecast_start = history_end
            forecast_end = forecast_start + self.config.forecast_steps

            history_df = df.iloc[history_start:history_end].copy()
            forecast_df = df.iloc[forecast_start:forecast_end].copy()

            # Create sample
            sample = self._create_sample(history_df, forecast_df)
            if sample is not None:
                samples.append(sample)

        return samples

    def _create_sample(self, history_df: pd.DataFrame,
                      forecast_df: pd.DataFrame) -> Optional[Dict]:
        """Create a single training sample."""
        try:
            # Basic sample info
            sample = {
                'mmsi': history_df['mmsi'].iloc[0],
                'start_time': history_df['timestamp'].iloc[0],
                'end_time': forecast_df['timestamp'].iloc[-1],
                'history_length': len(history_df),
                'forecast_length': len(forecast_df)
            }

            # Extract features
            feature_cols = ['lat', 'lon', 'sog', 'cog_sin', 'cog_cos']

            # History features (input)
            history_features = history_df[feature_cols].values.astype(np.float32)

            # Handle relative coordinates
            if self.config.use_relative_coords:
                # Use last history point as reference
                ref_lat = history_features[-1, 0]
                ref_lon = history_features[-1, 1]

                # Convert to relative positions (approximate)
                history_features[:, 0] = (history_features[:, 0] - ref_lat) * 111000  # meters
                history_features[:, 1] = (history_features[:, 1] - ref_lon) * 111000 * np.cos(np.deg2rad(ref_lat))

                sample['reference_lat'] = ref_lat
                sample['reference_lon'] = ref_lon

            sample['history_features'] = history_features

            # Forecast targets (output)
            forecast_positions = forecast_df[['lat', 'lon']].values.astype(np.float32)

            if self.config.use_relative_coords:
                # Convert forecast to relative coordinates
                forecast_positions[:, 0] = (forecast_positions[:, 0] - ref_lat) * 111000
                forecast_positions[:, 1] = (forecast_positions[:, 1] - ref_lon) * 111000 * np.cos(np.deg2rad(ref_lat))

            sample['forecast_positions'] = forecast_positions

            # Additional metadata
            sample['avg_sog'] = history_df['sog'].mean()
            sample['initial_cog'] = history_df['cog'].iloc[0]
            sample['final_cog'] = history_df['cog'].iloc[-1]
            sample['interpolation_ratio'] = history_df['is_interpolated'].mean()

            return sample

        except Exception as e:
            logger.warning(f"Failed to create sample: {e}")
            return None

    def slice_all_trajectories(self, df: pd.DataFrame) -> List[Dict]:
        """
        Slice all vessel trajectories into samples.

        Args:
            df: Multi-vessel resampled AIS DataFrame

        Returns:
            List of all trajectory samples
        """
        logger.info(f"Slicing trajectories for {df['mmsi'].nunique()} vessels")

        all_samples = []
        failed_vessels = []

        for mmsi, vessel_df in df.groupby('mmsi'):
            try:
                vessel_samples = self.slice_vessel_trajectory(vessel_df)
                all_samples.extend(vessel_samples)
            except Exception as e:
                logger.warning(f"Failed to slice vessel {mmsi}: {e}")
                failed_vessels.append(mmsi)

        logger.info(f"Generated {len(all_samples)} samples from "
                   f"{df['mmsi'].nunique() - len(failed_vessels)} vessels")

        if failed_vessels:
            logger.warning(f"Failed to process {len(failed_vessels)} vessels")

        return all_samples

    def split_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split samples into train/validation/test sets.

        Args:
            samples: List of trajectory samples

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        if self.config.split_by_mmsi:
            # Split by vessel to avoid data leakage
            unique_mmsi = list(set([sample['mmsi'] for sample in samples]))

            # First split: train vs (val + test)
            train_mmsi, temp_mmsi = train_test_split(
                unique_mmsi,
                test_size=(self.config.val_ratio + self.config.test_ratio),
                random_state=42
            )

            # Second split: val vs test
            val_ratio_adjusted = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
            val_mmsi, test_mmsi = train_test_split(
                temp_mmsi,
                test_size=(1 - val_ratio_adjusted),
                random_state=42
            )

            # Assign samples to splits
            train_samples = [s for s in samples if s['mmsi'] in train_mmsi]
            val_samples = [s for s in samples if s['mmsi'] in val_mmsi]
            test_samples = [s for s in samples if s['mmsi'] in test_mmsi]

        else:
            # Simple random split
            train_samples, temp_samples = train_test_split(
                samples,
                test_size=(self.config.val_ratio + self.config.test_ratio),
                random_state=42
            )

            val_ratio_adjusted = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
            val_samples, test_samples = train_test_split(
                temp_samples,
                test_size=(1 - val_ratio_adjusted),
                random_state=42
            )

        logger.info(f"Data split: {len(train_samples)} train, {len(val_samples)} val, "
                   f"{len(test_samples)} test samples")

        return train_samples, val_samples, test_samples

    def save_samples(self, samples: List[Dict], output_path: Path):
        """Save samples to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(samples, f)

        logger.info(f"Saved {len(samples)} samples to {output_path}")

    def load_samples(self, input_path: Path) -> List[Dict]:
        """Load samples from disk."""
        with open(input_path, 'rb') as f:
            samples = pickle.load(f)

        logger.info(f"Loaded {len(samples)} samples from {input_path}")
        return samples

    def get_slicing_stats(self, samples: List[Dict]) -> Dict:
        """Get statistics about sliced samples."""
        if not samples:
            return {}

        mmsi_counts = {}
        for sample in samples:
            mmsi = sample['mmsi']
            mmsi_counts[mmsi] = mmsi_counts.get(mmsi, 0) + 1

        avg_sogs = [sample['avg_sog'] for sample in samples]
        interp_ratios = [sample['interpolation_ratio'] for sample in samples]

        stats = {
            'total_samples': len(samples),
            'unique_vessels': len(mmsi_counts),
            'samples_per_vessel': {
                'mean': np.mean(list(mmsi_counts.values())),
                'std': np.std(list(mmsi_counts.values())),
                'min': min(mmsi_counts.values()),
                'max': max(mmsi_counts.values())
            },
            'avg_sog_distribution': {
                'mean': np.mean(avg_sogs),
                'std': np.std(avg_sogs),
                'min': np.min(avg_sogs),
                'max': np.max(avg_sogs)
            },
            'interpolation_stats': {
                'mean_ratio': np.mean(interp_ratios),
                'max_ratio': np.max(interp_ratios)
            },
            'time_coverage': {
                'start': min([sample['start_time'] for sample in samples]),
                'end': max([sample['end_time'] for sample in samples])
            }
        }

        return stats


def create_numpy_arrays(samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert samples to numpy arrays for model training.

    Args:
        samples: List of trajectory samples

    Returns:
        Tuple of (X, y) arrays where X is input features and y is target positions
    """
    if not samples:
        return np.array([]), np.array([])

    # Stack all samples
    X = np.stack([sample['history_features'] for sample in samples])
    y = np.stack([sample['forecast_positions'] for sample in samples])

    logger.info(f"Created arrays: X shape {X.shape}, y shape {y.shape}")
    return X, y


if __name__ == "__main__":
    # Example usage
    import yaml

    # Load configuration
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize slicer
    slicer = TrajectorySlicer(config)

    # Process example data
    # df_resampled = pd.read_parquet("data/interim/resampled_ais.parquet")
    # samples = slicer.slice_all_trajectories(df_resampled)
    # train_samples, val_samples, test_samples = slicer.split_samples(samples)
    # stats = slicer.get_slicing_stats(samples)
    # print(stats)
