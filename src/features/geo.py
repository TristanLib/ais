"""
Geographic feature engineering for ship trajectory data.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Optional
from geopy.distance import geodesic
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class GeoFeatureEngine:
    """Geographic feature engineering utilities."""

    def __init__(self):
        """Initialize geographic feature engine."""
        self.earth_radius_km = 6371.0
        self.nautical_mile_to_km = 1.852

    def haversine_distance(self, lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two points in kilometers.

        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)

        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return self.earth_radius_km * c

    def haversine_distance_vectorized(self, coords1: np.ndarray,
                                    coords2: np.ndarray) -> np.ndarray:
        """
        Vectorized haversine distance calculation.

        Args:
            coords1: Array of shape (N, 2) with [lat, lon] in degrees
            coords2: Array of shape (N, 2) with [lat, lon] in degrees

        Returns:
            Array of distances in kilometers
        """
        lat1, lon1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
        lat2, lon2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        return self.earth_radius_km * c

    def bearing_between_points(self, lat1: float, lon1: float,
                             lat2: float, lon2: float) -> float:
        """
        Calculate bearing between two points in degrees.

        Args:
            lat1, lon1: Start point coordinates (degrees)
            lat2, lon2: End point coordinates (degrees)

        Returns:
            Bearing in degrees (0-360)
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        bearing = np.arctan2(y, x)
        bearing = np.degrees(bearing)
        bearing = (bearing + 360) % 360

        return bearing

    def calculate_speed_over_ground(self, trajectory: pd.DataFrame) -> pd.Series:
        """
        Calculate speed over ground from position data.

        Args:
            trajectory: DataFrame with 'timestamp', 'lat', 'lon' columns

        Returns:
            Series of SOG values in knots
        """
        if len(trajectory) < 2:
            return pd.Series([], dtype=float)

        # Calculate distances
        coords = trajectory[['lat', 'lon']].values
        distances_km = self.haversine_distance_vectorized(coords[:-1], coords[1:])

        # Calculate time differences
        time_diffs = trajectory['timestamp'].diff().dt.total_seconds().iloc[1:].values

        # Calculate speeds (km/h to knots)
        speeds_kmh = distances_km / (time_diffs / 3600)
        speeds_knots = speeds_kmh / self.nautical_mile_to_km

        # Prepend NaN for first point
        speeds = np.concatenate([[np.nan], speeds_knots])

        return pd.Series(speeds, index=trajectory.index)

    def calculate_course_over_ground(self, trajectory: pd.DataFrame) -> pd.Series:
        """
        Calculate course over ground from position data.

        Args:
            trajectory: DataFrame with 'lat', 'lon' columns

        Returns:
            Series of COG values in degrees
        """
        if len(trajectory) < 2:
            return pd.Series([], dtype=float)

        coords = trajectory[['lat', 'lon']].values

        # Calculate bearings between consecutive points
        bearings = []
        for i in range(len(coords) - 1):
            bearing = self.bearing_between_points(
                coords[i, 0], coords[i, 1],
                coords[i+1, 0], coords[i+1, 1]
            )
            bearings.append(bearing)

        # Prepend NaN for first point
        cogs = np.concatenate([[np.nan], bearings])

        return pd.Series(cogs, index=trajectory.index)

    def calculate_turn_rate(self, cog_series: pd.Series, dt_seconds: float = 60) -> pd.Series:
        """
        Calculate turn rate from course over ground.

        Args:
            cog_series: Series of COG values in degrees
            dt_seconds: Time step in seconds

        Returns:
            Series of turn rates in degrees per minute
        """
        # Calculate angular differences (handling wrap-around)
        cog_diff = cog_series.diff()

        # Handle wrap-around at 0/360 degrees
        cog_diff = np.where(cog_diff > 180, cog_diff - 360, cog_diff)
        cog_diff = np.where(cog_diff < -180, cog_diff + 360, cog_diff)

        # Convert to degrees per minute
        turn_rate = cog_diff / (dt_seconds / 60)

        return turn_rate

    def calculate_acceleration(self, sog_series: pd.Series, dt_seconds: float = 60) -> pd.Series:
        """
        Calculate acceleration from speed over ground.

        Args:
            sog_series: Series of SOG values in knots
            dt_seconds: Time step in seconds

        Returns:
            Series of accelerations in knots per minute
        """
        sog_diff = sog_series.diff()
        acceleration = sog_diff / (dt_seconds / 60)

        return acceleration

    def add_kinematic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add kinematic features to trajectory data.

        Args:
            df: DataFrame with trajectory data

        Returns:
            DataFrame with additional kinematic features
        """
        df = df.copy()

        # Group by vessel and calculate features
        for mmsi, vessel_df in df.groupby('mmsi'):
            vessel_df = vessel_df.sort_values('timestamp')

            # Calculate derived features if not present
            if 'sog' not in vessel_df.columns:
                df.loc[vessel_df.index, 'sog'] = self.calculate_speed_over_ground(vessel_df)

            if 'cog' not in vessel_df.columns:
                df.loc[vessel_df.index, 'cog'] = self.calculate_course_over_ground(vessel_df)

            # Calculate turn rate
            if 'cog' in vessel_df.columns:
                df.loc[vessel_df.index, 'turn_rate'] = self.calculate_turn_rate(vessel_df['cog'])

            # Calculate acceleration
            if 'sog' in vessel_df.columns:
                df.loc[vessel_df.index, 'acceleration'] = self.calculate_acceleration(vessel_df['sog'])

        return df

    def find_nearby_vessels(self, target_position: Tuple[float, float],
                           all_positions: Dict[int, Tuple[float, float]],
                           radius_nmi: float = 2.0,
                           max_neighbors: int = 5) -> List[int]:
        """
        Find nearby vessels within a given radius.

        Args:
            target_position: (lat, lon) of target vessel
            all_positions: Dict of {mmsi: (lat, lon)} for all vessels
            radius_nmi: Search radius in nautical miles
            max_neighbors: Maximum number of neighbors to return

        Returns:
            List of MMSI values for nearby vessels
        """
        target_lat, target_lon = target_position
        distances = []

        for mmsi, (lat, lon) in all_positions.items():
            distance_km = self.haversine_distance(target_lat, target_lon, lat, lon)
            distance_nmi = distance_km / self.nautical_mile_to_km

            if distance_nmi <= radius_nmi:
                distances.append((mmsi, distance_nmi))

        # Sort by distance and return top neighbors
        distances.sort(key=lambda x: x[1])
        neighbors = [mmsi for mmsi, _ in distances[:max_neighbors]]

        return neighbors

    def calculate_relative_motion(self, target_trajectory: pd.DataFrame,
                                neighbor_trajectory: pd.DataFrame) -> Dict:
        """
        Calculate relative motion parameters between two vessels.

        Args:
            target_trajectory: Target vessel trajectory
            neighbor_trajectory: Neighbor vessel trajectory

        Returns:
            Dictionary with relative motion parameters
        """
        # Find common time points
        common_times = set(target_trajectory['timestamp']).intersection(
            set(neighbor_trajectory['timestamp'])
        )

        if not common_times:
            return {}

        # Get trajectories at common times
        target_common = target_trajectory[target_trajectory['timestamp'].isin(common_times)].copy()
        neighbor_common = neighbor_trajectory[neighbor_trajectory['timestamp'].isin(common_times)].copy()

        if len(target_common) == 0 or len(neighbor_common) == 0:
            return {}

        # Calculate relative positions and velocities
        target_pos = target_common[['lat', 'lon']].values
        neighbor_pos = neighbor_common[['lat', 'lon']].values

        # Relative distance
        rel_distances = self.haversine_distance_vectorized(target_pos, neighbor_pos)

        # Relative bearing
        rel_bearings = []
        for i in range(len(target_pos)):
            bearing = self.bearing_between_points(
                target_pos[i, 0], target_pos[i, 1],
                neighbor_pos[i, 0], neighbor_pos[i, 1]
            )
            rel_bearings.append(bearing)

        return {
            'min_distance_nmi': np.min(rel_distances) / self.nautical_mile_to_km,
            'avg_distance_nmi': np.mean(rel_distances) / self.nautical_mile_to_km,
            'distance_trend': np.polyfit(range(len(rel_distances)), rel_distances, 1)[0],
            'avg_bearing': np.mean(rel_bearings),
            'bearing_std': np.std(rel_bearings)
        }

    def create_spatial_grid(self, df: pd.DataFrame,
                           grid_size_km: float = 10.0) -> pd.DataFrame:
        """
        Create spatial grid features for trajectory data.

        Args:
            df: Trajectory DataFrame
            grid_size_km: Grid cell size in kilometers

        Returns:
            DataFrame with grid features added
        """
        df = df.copy()

        # Calculate grid coordinates
        lat_min, lat_max = df['lat'].min(), df['lat'].max()
        lon_min, lon_max = df['lon'].min(), df['lon'].max()

        # Approximate grid size in degrees
        lat_grid_size = grid_size_km / 111.0  # 1 degree ≈ 111 km
        lon_grid_size = grid_size_km / (111.0 * np.cos(np.deg2rad((lat_min + lat_max) / 2)))

        # Assign grid cells
        df['grid_lat'] = ((df['lat'] - lat_min) // lat_grid_size).astype(int)
        df['grid_lon'] = ((df['lon'] - lon_min) // lon_grid_size).astype(int)
        df['grid_id'] = df['grid_lat'] * 1000 + df['grid_lon']  # Unique grid ID

        return df


def convert_to_utm(df: pd.DataFrame, utm_zone: Optional[int] = None) -> pd.DataFrame:
    """
    Convert lat/lon coordinates to UTM projection.

    Args:
        df: DataFrame with 'lat', 'lon' columns
        utm_zone: UTM zone number (auto-detected if None)

    Returns:
        DataFrame with 'utm_x', 'utm_y' columns added
    """
    try:
        import pyproj

        # Auto-detect UTM zone if not provided
        if utm_zone is None:
            avg_lon = df['lon'].mean()
            utm_zone = int((avg_lon + 180) / 6) + 1

        # Create UTM projection
        utm_crs = pyproj.CRS.from_epsg(32600 + utm_zone)  # UTM North
        transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

        # Transform coordinates
        utm_x, utm_y = transformer.transform(df['lon'].values, df['lat'].values)

        df = df.copy()
        df['utm_x'] = utm_x
        df['utm_y'] = utm_y
        df['utm_zone'] = utm_zone

        logger.info(f"Converted to UTM zone {utm_zone}")
        return df

    except ImportError:
        logger.warning("pyproj not available, using approximate conversion")
        return convert_to_local_meters(df)


def convert_to_local_meters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert lat/lon to approximate local meter coordinates.

    Args:
        df: DataFrame with 'lat', 'lon' columns

    Returns:
        DataFrame with 'x_m', 'y_m' columns (meters from reference point)
    """
    df = df.copy()

    # Use center of data as reference point
    ref_lat = df['lat'].mean()
    ref_lon = df['lon'].mean()

    # Approximate conversion (good for small areas)
    df['x_m'] = (df['lon'] - ref_lon) * 111000 * np.cos(np.deg2rad(ref_lat))
    df['y_m'] = (df['lat'] - ref_lat) * 111000

    df['ref_lat'] = ref_lat
    df['ref_lon'] = ref_lon

    return df


if __name__ == "__main__":
    # Example usage
    geo_engine = GeoFeatureEngine()

    # Test distance calculation
    dist = geo_engine.haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
    print(f"Distance NYC to LA: {dist:.2f} km")

    # Test bearing calculation
    bearing = geo_engine.bearing_between_points(40.7128, -74.0060, 34.0522, -118.2437)
    print(f"Bearing NYC to LA: {bearing:.2f} degrees")
