"""
Baseline models for ship trajectory prediction.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseTrajectoryPredictor(ABC):
    """Base class for trajectory prediction models."""

    def __init__(self, config: Dict):
        """Initialize predictor with configuration."""
        self.config = config
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict future trajectories."""
        pass

    def get_params(self) -> Dict:
        """Get model parameters."""
        return {}


class ConstantVelocityPredictor(BaseTrajectoryPredictor):
    """Constant velocity (CV) baseline predictor."""

    def __init__(self, config: Dict):
        """Initialize CV predictor."""
        super().__init__(config)
        self.dt_minutes = config.get('dt_minutes', 1)
        self.forecast_steps = config.get('forecast_steps', 30)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """CV model doesn't require fitting."""
        self.is_fitted = True
        logger.info("CV model fitted (no parameters to learn)")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using constant velocity assumption.

        Args:
            X: Input sequences of shape (batch_size, seq_len, features)
               Features expected: [lat, lon, sog, cog_sin, cog_cos] or [x, y, sog, cog_sin, cog_cos]

        Returns:
            Predicted positions of shape (batch_size, forecast_steps, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        batch_size = X.shape[0]
        predictions = np.zeros((batch_size, self.forecast_steps, 2))

        for i in range(batch_size):
            # Get last two positions to estimate velocity
            if X.shape[1] >= 2:
                pos_current = X[i, -1, :2]  # [lat/x, lon/y]
                pos_previous = X[i, -2, :2]

                # Calculate velocity (position change per time step)
                velocity = pos_current - pos_previous
            else:
                # If only one position, use SOG and COG
                pos_current = X[i, -1, :2]
                sog = X[i, -1, 2]  # Speed over ground
                cog_sin = X[i, -1, 3]
                cog_cos = X[i, -1, 4]

                # Convert SOG (knots) to position change per minute
                # Approximate: 1 knot ≈ 0.0003 degrees latitude per minute
                speed_deg_per_min = sog * 0.0003

                velocity = np.array([
                    speed_deg_per_min * cog_sin,  # North component
                    speed_deg_per_min * cog_cos   # East component
                ])

            # Predict future positions
            for t in range(self.forecast_steps):
                predictions[i, t, :] = pos_current + velocity * (t + 1)

        return predictions


class ConstantAccelerationPredictor(BaseTrajectoryPredictor):
    """Constant acceleration (CA) baseline predictor."""

    def __init__(self, config: Dict):
        """Initialize CA predictor."""
        super().__init__(config)
        self.dt_minutes = config.get('dt_minutes', 1)
        self.forecast_steps = config.get('forecast_steps', 30)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """CA model doesn't require fitting."""
        self.is_fitted = True
        logger.info("CA model fitted (no parameters to learn)")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using constant acceleration assumption.

        Args:
            X: Input sequences of shape (batch_size, seq_len, features)

        Returns:
            Predicted positions of shape (batch_size, forecast_steps, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        batch_size = X.shape[0]
        predictions = np.zeros((batch_size, self.forecast_steps, 2))

        for i in range(batch_size):
            seq_len = X.shape[1]

            if seq_len >= 3:
                # Use last three positions to estimate acceleration
                pos_current = X[i, -1, :2]
                pos_prev1 = X[i, -2, :2]
                pos_prev2 = X[i, -3, :2]

                # Calculate velocity and acceleration
                velocity_current = pos_current - pos_prev1
                velocity_previous = pos_prev1 - pos_prev2
                acceleration = velocity_current - velocity_previous

            elif seq_len >= 2:
                # Fall back to constant velocity
                pos_current = X[i, -1, :2]
                pos_previous = X[i, -2, :2]
                velocity_current = pos_current - pos_previous
                acceleration = np.zeros(2)

            else:
                # Fall back to stationary prediction
                pos_current = X[i, -1, :2]
                velocity_current = np.zeros(2)
                acceleration = np.zeros(2)

            # Predict future positions using kinematic equation
            for t in range(self.forecast_steps):
                dt = t + 1
                predictions[i, t, :] = (pos_current +
                                      velocity_current * dt +
                                      0.5 * acceleration * dt**2)

        return predictions


class LinearExtrapolationPredictor(BaseTrajectoryPredictor):
    """Linear extrapolation predictor using least squares fit."""

    def __init__(self, config: Dict):
        """Initialize linear extrapolation predictor."""
        super().__init__(config)
        self.forecast_steps = config.get('forecast_steps', 30)
        self.window_size = config.get('extrapolation_window', 10)  # Use last N points for fitting

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Linear extrapolation doesn't require global fitting."""
        self.is_fitted = True
        logger.info("Linear extrapolation model fitted")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using linear extrapolation of recent trajectory.

        Args:
            X: Input sequences of shape (batch_size, seq_len, features)

        Returns:
            Predicted positions of shape (batch_size, forecast_steps, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        batch_size, seq_len = X.shape[:2]
        predictions = np.zeros((batch_size, self.forecast_steps, 2))

        for i in range(batch_size):
            # Use last window_size points for linear fit
            window = min(self.window_size, seq_len)
            positions = X[i, -window:, :2]  # [lat/x, lon/y]

            if window >= 2:
                # Fit linear trend
                t = np.arange(window)

                # Fit for each coordinate
                for coord in range(2):
                    # Linear least squares: y = a*t + b
                    A = np.vstack([t, np.ones(len(t))]).T
                    coeffs, _, _, _ = np.linalg.lstsq(A, positions[:, coord], rcond=None)
                    slope, intercept = coeffs

                    # Extrapolate
                    future_t = np.arange(window, window + self.forecast_steps)
                    predictions[i, :, coord] = slope * future_t + intercept
            else:
                # Fall back to constant position
                predictions[i, :, :] = positions[-1, :]

        return predictions


class MovingAveragePredictor(BaseTrajectoryPredictor):
    """Moving average predictor with trend estimation."""

    def __init__(self, config: Dict):
        """Initialize moving average predictor."""
        super().__init__(config)
        self.forecast_steps = config.get('forecast_steps', 30)
        self.window_size = config.get('ma_window', 5)
        self.trend_weight = config.get('trend_weight', 0.5)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Moving average doesn't require global fitting."""
        self.is_fitted = True
        logger.info("Moving average model fitted")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using moving average with trend.

        Args:
            X: Input sequences of shape (batch_size, seq_len, features)

        Returns:
            Predicted positions of shape (batch_size, forecast_steps, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        batch_size, seq_len = X.shape[:2]
        predictions = np.zeros((batch_size, self.forecast_steps, 2))

        for i in range(batch_size):
            positions = X[i, :, :2]  # [lat/x, lon/y]

            # Calculate moving average of recent positions
            window = min(self.window_size, seq_len)
            avg_position = np.mean(positions[-window:], axis=0)

            # Estimate trend from recent changes
            if seq_len >= 2:
                recent_changes = positions[1:] - positions[:-1]
                avg_trend = np.mean(recent_changes[-window:], axis=0)
            else:
                avg_trend = np.zeros(2)

            # Predict future positions
            for t in range(self.forecast_steps):
                predictions[i, t, :] = avg_position + self.trend_weight * avg_trend * (t + 1)

        return predictions


def create_baseline_predictor(model_name: str, config: Dict) -> BaseTrajectoryPredictor:
    """
    Factory function to create baseline predictors.

    Args:
        model_name: Name of the baseline model
        config: Configuration dictionary

    Returns:
        Baseline predictor instance
    """
    predictors = {
        'cv': ConstantVelocityPredictor,
        'constant_velocity': ConstantVelocityPredictor,
        'ca': ConstantAccelerationPredictor,
        'constant_acceleration': ConstantAccelerationPredictor,
        'linear': LinearExtrapolationPredictor,
        'linear_extrapolation': LinearExtrapolationPredictor,
        'ma': MovingAveragePredictor,
        'moving_average': MovingAveragePredictor
    }

    if model_name.lower() not in predictors:
        raise ValueError(f"Unknown baseline model: {model_name}. "
                        f"Available: {list(predictors.keys())}")

    predictor_class = predictors[model_name.lower()]
    return predictor_class(config)


class BaselineEnsemble:
    """Ensemble of baseline predictors."""

    def __init__(self, predictors: List[BaseTrajectoryPredictor], weights: Optional[List[float]] = None):
        """
        Initialize ensemble.

        Args:
            predictors: List of baseline predictors
            weights: Optional weights for ensemble (uniform if None)
        """
        self.predictors = predictors
        self.weights = weights or [1.0 / len(predictors)] * len(predictors)

        if len(self.weights) != len(self.predictors):
            raise ValueError("Number of weights must match number of predictors")

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all predictors in ensemble."""
        for predictor in self.predictors:
            predictor.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted ensemble."""
        predictions = []

        for predictor in self.predictors:
            pred = predictor.predict(X)
            predictions.append(pred)

        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred

        return ensemble_pred


if __name__ == "__main__":
    # Example usage
    config = {
        'dt_minutes': 1,
        'forecast_steps': 30,
        'extrapolation_window': 10,
        'ma_window': 5,
        'trend_weight': 0.5
    }

    # Create sample data
    batch_size, seq_len, n_features = 10, 60, 5
    X = np.random.randn(batch_size, seq_len, n_features)
    y = np.random.randn(batch_size, 30, 2)

    # Test baseline models
    models = ['cv', 'ca', 'linear', 'ma']

    for model_name in models:
        print(f"\nTesting {model_name} predictor:")
        predictor = create_baseline_predictor(model_name, config)
        predictor.fit(X, y)
        predictions = predictor.predict(X)
        print(f"Predictions shape: {predictions.shape}")

    # Test ensemble
    print("\nTesting ensemble:")
    predictors = [create_baseline_predictor(name, config) for name in models]
    ensemble = BaselineEnsemble(predictors)
    ensemble.fit(X, y)
    ensemble_pred = ensemble.predict(X)
    print(f"Ensemble predictions shape: {ensemble_pred.shape}")
