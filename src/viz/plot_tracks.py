"""
Visualization tools for ship trajectories and predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """Visualizer for ship trajectories and predictions."""

    def __init__(self, config: Dict):
        """Initialize trajectory visualizer."""
        self.config = config
        viz_config = config.get('visualization', {})

        # Plot settings
        self.figure_size = viz_config.get('figure_size', [12, 8])
        self.dpi = viz_config.get('dpi', 300)

        # Colors
        colors = viz_config.get('trajectory_colors', {})
        self.color_actual = colors.get('actual', 'blue')
        self.color_predicted = colors.get('predicted', 'red')
        self.color_neighbors = colors.get('neighbors', 'gray')
        self.color_risk_zone = colors.get('risk_zone', 'orange')

        # Map settings
        self.map_style = viz_config.get('map_style', 'OpenStreetMap')

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_single_trajectory(self, actual_positions: np.ndarray,
                             predicted_positions: Optional[np.ndarray] = None,
                             timestamps: Optional[List] = None,
                             title: str = "Ship Trajectory",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a single ship trajectory with actual and predicted positions.

        Args:
            actual_positions: Actual positions (N, 2) - [lat, lon] or [x, y]
            predicted_positions: Predicted positions (M, 2)
            timestamps: Optional timestamps for the trajectory
            title: Plot title
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        # Plot actual trajectory
        ax.plot(actual_positions[:, 1], actual_positions[:, 0],
                color=self.color_actual, linewidth=2, marker='o', markersize=4,
                label='Actual Trajectory', alpha=0.8)

        # Mark start and end points
        ax.scatter(actual_positions[0, 1], actual_positions[0, 0],
                  color='green', s=100, marker='s', label='Start', zorder=5)
        ax.scatter(actual_positions[-1, 1], actual_positions[-1, 0],
                  color='red', s=100, marker='X', label='End', zorder=5)

        # Plot predicted trajectory if provided
        if predicted_positions is not None:
            # Connect last actual position to first predicted position
            connection_x = [actual_positions[-1, 1], predicted_positions[0, 1]]
            connection_y = [actual_positions[-1, 0], predicted_positions[0, 0]]
            ax.plot(connection_x, connection_y,
                   color=self.color_predicted, linestyle='--', alpha=0.5)

            ax.plot(predicted_positions[:, 1], predicted_positions[:, 0],
                    color=self.color_predicted, linewidth=2, marker='o', markersize=4,
                    label='Predicted Trajectory', alpha=0.8, linestyle='--')

            ax.scatter(predicted_positions[-1, 1], predicted_positions[-1, 0],
                      color=self.color_predicted, s=100, marker='*',
                      label='Predicted End', zorder=5)

        ax.set_xlabel('Longitude' if np.max(np.abs(actual_positions[:, 1])) < 1000 else 'X (meters)')
        ax.set_ylabel('Latitude' if np.max(np.abs(actual_positions[:, 0])) < 1000 else 'Y (meters)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Add direction arrows
        self._add_direction_arrows(ax, actual_positions, self.color_actual)
        if predicted_positions is not None:
            self._add_direction_arrows(ax, predicted_positions, self.color_predicted)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved trajectory plot to {save_path}")

        return fig

    def _add_direction_arrows(self, ax, positions: np.ndarray, color: str,
                            arrow_spacing: int = 5):
        """Add direction arrows to trajectory."""
        if len(positions) < 2:
            return

        for i in range(0, len(positions) - 1, arrow_spacing):
            if i + 1 >= len(positions):
                break

            dx = positions[i + 1, 1] - positions[i, 1]
            dy = positions[i + 1, 0] - positions[i, 0]

            if abs(dx) > 1e-6 or abs(dy) > 1e-6:  # Avoid zero-length arrows
                ax.arrow(positions[i, 1], positions[i, 0], dx * 0.3, dy * 0.3,
                        head_width=0.001, head_length=0.001, fc=color, ec=color,
                        alpha=0.6, length_includes_head=True)

    def plot_multiple_trajectories(self, trajectories: List[Dict],
                                 title: str = "Multiple Ship Trajectories",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple ship trajectories.

        Args:
            trajectories: List of trajectory dictionaries with keys:
                         'actual', 'predicted' (optional), 'mmsi', 'label' (optional)
            title: Plot title
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

        for i, traj in enumerate(trajectories):
            actual = traj['actual']
            predicted = traj.get('predicted')
            mmsi = traj.get('mmsi', f'Vessel {i+1}')
            label = traj.get('label', f'MMSI {mmsi}')
            color = colors[i]

            # Plot actual trajectory
            ax.plot(actual[:, 1], actual[:, 0],
                   color=color, linewidth=2, marker='o', markersize=3,
                   label=f'{label} (Actual)', alpha=0.8)

            # Plot predicted trajectory if available
            if predicted is not None:
                ax.plot(predicted[:, 1], predicted[:, 0],
                       color=color, linewidth=2, marker='s', markersize=3,
                       label=f'{label} (Predicted)', alpha=0.6, linestyle='--')

            # Mark start points
            ax.scatter(actual[0, 1], actual[0, 0],
                      color=color, s=50, marker='o', alpha=0.8, zorder=5)

        ax.set_xlabel('Longitude' if np.max(np.abs(trajectories[0]['actual'][:, 1])) < 1000 else 'X (meters)')
        ax.set_ylabel('Latitude' if np.max(np.abs(trajectories[0]['actual'][:, 0])) < 1000 else 'Y (meters)')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved multiple trajectories plot to {save_path}")

        return fig

    def plot_prediction_errors(self, predictions: np.ndarray, targets: np.ndarray,
                             horizons: List[int] = [5, 10, 20, 30],
                             title: str = "Prediction Errors",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot prediction error analysis.

        Args:
            predictions: Model predictions (N, T, 2)
            targets: Ground truth (N, T, 2)
            horizons: Time horizons to analyze (minutes)
            title: Plot title
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Calculate errors
        errors = np.linalg.norm(predictions - targets, axis=-1)  # (N, T)

        # 1. Error distribution
        ax = axes[0, 0]
        ax.hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Prediction Error (meters)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.grid(True, alpha=0.3)

        # 2. Error vs prediction horizon
        ax = axes[0, 1]
        mean_errors = np.mean(errors, axis=0)
        std_errors = np.std(errors, axis=0)
        time_steps = np.arange(len(mean_errors))

        ax.plot(time_steps, mean_errors, 'b-', linewidth=2, label='Mean Error')
        ax.fill_between(time_steps,
                       mean_errors - std_errors,
                       mean_errors + std_errors,
                       alpha=0.3, label='±1 STD')

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Prediction Error (meters)')
        ax.set_title('Error vs Prediction Horizon')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Error heatmap by sample and time
        ax = axes[1, 0]
        sample_indices = np.random.choice(len(errors), min(50, len(errors)), replace=False)
        error_subset = errors[sample_indices]

        im = ax.imshow(error_subset, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Sample Index')
        ax.set_title('Error Heatmap (Sample)')
        plt.colorbar(im, ax=ax, label='Error (meters)')

        # 4. Cumulative error distribution
        ax = axes[1, 1]
        flat_errors = errors.flatten()
        sorted_errors = np.sort(flat_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        ax.plot(sorted_errors, cumulative, 'g-', linewidth=2)
        ax.set_xlabel('Prediction Error (meters)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Error Distribution')
        ax.grid(True, alpha=0.3)

        # Add percentile lines
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            error_p = np.percentile(flat_errors, p)
            ax.axvline(x=error_p, color='red', linestyle='--', alpha=0.7)
            ax.text(error_p, p/100, f'{p}%', rotation=90, va='bottom')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved prediction error plot to {save_path}")

        return fig

    def create_interactive_map(self, trajectories: List[Dict],
                             center_lat: float = 40.7128, center_lon: float = -74.0060,
                             save_path: Optional[str] = None) -> folium.Map:
        """
        Create interactive map with trajectories.

        Args:
            trajectories: List of trajectory dictionaries
            center_lat: Map center latitude
            center_lon: Map center longitude
            save_path: Path to save the HTML map

        Returns:
            Folium map object
        """
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles=self.map_style
        )

        colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred',
                 'lightred', 'beige', 'darkblue', 'darkgreen']

        for i, traj in enumerate(trajectories):
            actual = traj['actual']
            predicted = traj.get('predicted')
            mmsi = traj.get('mmsi', f'Vessel {i+1}')
            color = colors[i % len(colors)]

            # Add actual trajectory
            actual_coords = [[pos[0], pos[1]] for pos in actual]
            folium.PolyLine(
                actual_coords,
                color=color,
                weight=3,
                opacity=0.8,
                popup=f'MMSI {mmsi} - Actual'
            ).add_to(m)

            # Add start and end markers
            folium.Marker(
                [actual[0, 0], actual[0, 1]],
                popup=f'MMSI {mmsi} - Start',
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)

            folium.Marker(
                [actual[-1, 0], actual[-1, 1]],
                popup=f'MMSI {mmsi} - End',
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)

            # Add predicted trajectory if available
            if predicted is not None:
                pred_coords = [[pos[0], pos[1]] for pos in predicted]
                folium.PolyLine(
                    pred_coords,
                    color=color,
                    weight=2,
                    opacity=0.6,
                    dash_array='10, 10',
                    popup=f'MMSI {mmsi} - Predicted'
                ).add_to(m)

                folium.Marker(
                    [predicted[-1, 0], predicted[-1, 1]],
                    popup=f'MMSI {mmsi} - Predicted End',
                    icon=folium.Icon(color='orange', icon='star')
                ).add_to(m)

        # Add legend
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 150px; height: 90px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 10px">
        <p><b>Legend</b></p>
        <p><i class="fa fa-play" style="color:green"></i> Start</p>
        <p><i class="fa fa-stop" style="color:red"></i> End</p>
        <p><i class="fa fa-star" style="color:orange"></i> Predicted End</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        if save_path:
            m.save(save_path)
            logger.info(f"Saved interactive map to {save_path}")

        return m

    def plot_collision_scenario(self, vessels: List[Dict], risk_zones: List[Dict] = None,
                              title: str = "Collision Avoidance Scenario",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot collision avoidance scenario with risk zones.

        Args:
            vessels: List of vessel dictionaries with trajectory data
            risk_zones: List of risk zone dictionaries
            title: Plot title
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        colors = plt.cm.Set1(np.linspace(0, 1, len(vessels)))

        for i, vessel in enumerate(vessels):
            positions = vessel['positions']  # Nx2 array
            mmsi = vessel.get('mmsi', f'Vessel {i+1}')
            color = colors[i]

            # Plot trajectory
            ax.plot(positions[:, 0], positions[:, 1],
                   color=color, linewidth=2, marker='o', markersize=4,
                   label=f'MMSI {mmsi}', alpha=0.8)

            # Mark current position
            current_pos = positions[-1]
            ax.scatter(current_pos[0], current_pos[1],
                      color=color, s=100, marker='s', zorder=5)

            # Add vessel direction arrow
            if len(positions) > 1:
                prev_pos = positions[-2]
                dx = current_pos[0] - prev_pos[0]
                dy = current_pos[1] - prev_pos[1]

                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    ax.arrow(current_pos[0], current_pos[1], dx * 2, dy * 2,
                            head_width=50, head_length=50, fc=color, ec=color,
                            alpha=0.7, length_includes_head=True, zorder=6)

        # Plot risk zones
        if risk_zones:
            for risk_zone in risk_zones:
                center = risk_zone['center']  # [x, y]
                radius = risk_zone['radius']  # meters
                risk_level = risk_zone.get('risk_level', 'medium')

                # Choose color based on risk level
                if risk_level == 'critical':
                    zone_color = 'red'
                    alpha = 0.4
                elif risk_level == 'high':
                    zone_color = 'orange'
                    alpha = 0.3
                elif risk_level == 'medium':
                    zone_color = 'yellow'
                    alpha = 0.2
                else:
                    zone_color = 'green'
                    alpha = 0.1

                circle = plt.Circle(center, radius,
                                  fill=True, color=zone_color, alpha=alpha,
                                  label=f'Risk Zone ({risk_level})')
                ax.add_patch(circle)

        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved collision scenario plot to {save_path}")

        return fig


if __name__ == "__main__":
    # Example usage
    config = {
        'visualization': {
            'figure_size': [12, 8],
            'dpi': 300,
            'trajectory_colors': {
                'actual': 'blue',
                'predicted': 'red',
                'neighbors': 'gray',
                'risk_zone': 'orange'
            }
        }
    }

    # Create visualizer
    viz = TrajectoryVisualizer(config)

    # Generate sample data
    t = np.linspace(0, 2*np.pi, 50)
    actual_positions = np.column_stack([
        40.7128 + 0.01 * np.sin(t),  # lat
        -74.0060 + 0.01 * np.cos(t)  # lon
    ])

    predicted_positions = np.column_stack([
        40.7128 + 0.01 * np.sin(t[-30:] + 0.1),
        -74.0060 + 0.01 * np.cos(t[-30:] + 0.1)
    ])

    # Test single trajectory plot
    fig = viz.plot_single_trajectory(
        actual_positions, predicted_positions,
        title="Test Trajectory"
    )
    plt.show()

    print("Visualization example completed")
