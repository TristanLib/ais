"""
Physics-Informed Neural Network (PINN) model for ship trajectory prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PhysicsModule(nn.Module):
    """Physics constraints for ship motion."""

    def __init__(self, dt: float = 1.0):
        """
        Initialize physics module.

        Args:
            dt: Time step in minutes
        """
        super(PhysicsModule, self).__init__()

        self.dt = dt  # Time step

        # Physical constraints
        self.max_acceleration = 0.1  # Maximum acceleration (nm/min²)
        self.max_turn_rate = 5.0     # Maximum turn rate (degrees/min)
        self.max_speed = 30.0        # Maximum speed (knots)
        self.min_speed = 0.0         # Minimum speed (knots)

    def kinematic_constraint(self, positions: torch.Tensor,
                           velocities: torch.Tensor) -> torch.Tensor:
        """
        Kinematic constraint: v = dp/dt

        Args:
            positions: Position tensor (batch_size, seq_len, 2)
            velocities: Velocity tensor (batch_size, seq_len, 2)

        Returns:
            Kinematic residual
        """
        if positions.size(1) < 2:
            return torch.tensor(0.0, device=positions.device)

        # Compute numerical derivative of positions
        dp_dt = (positions[:, 1:, :] - positions[:, :-1, :]) / self.dt

        # Compare with provided velocities
        v_pred = velocities[:, :-1, :]  # Use velocity at previous time step

        # Kinematic residual
        residual = torch.mean((dp_dt - v_pred) ** 2)

        return residual

    def acceleration_constraint(self, velocities: torch.Tensor) -> torch.Tensor:
        """
        Acceleration constraint: limit maximum acceleration

        Args:
            velocities: Velocity tensor (batch_size, seq_len, 2)

        Returns:
            Acceleration constraint residual
        """
        if velocities.size(1) < 2:
            return torch.tensor(0.0, device=velocities.device)

        # Compute acceleration
        dv_dt = (velocities[:, 1:, :] - velocities[:, :-1, :]) / self.dt

        # Acceleration magnitude
        acceleration_mag = torch.norm(dv_dt, dim=-1)  # (batch_size, seq_len-1)

        # Penalize excessive acceleration
        max_acc_tensor = torch.tensor(self.max_acceleration, device=velocities.device)
        excess_acceleration = F.relu(acceleration_mag - max_acc_tensor)

        residual = torch.mean(excess_acceleration ** 2)

        return residual

    def turning_constraint(self, velocities: torch.Tensor) -> torch.Tensor:
        """
        Turning constraint: limit maximum turn rate

        Args:
            velocities: Velocity tensor (batch_size, seq_len, 2)

        Returns:
            Turning constraint residual
        """
        if velocities.size(1) < 2:
            return torch.tensor(0.0, device=velocities.device)

        # Compute heading angles
        headings = torch.atan2(velocities[:, :, 1], velocities[:, :, 0])  # (batch_size, seq_len)

        # Compute turn rates
        dheading_dt = headings[:, 1:] - headings[:, :-1]

        # Handle angle wrapping
        dheading_dt = torch.remainder(dheading_dt + math.pi, 2 * math.pi) - math.pi

        # Convert to degrees per minute
        turn_rate = torch.abs(dheading_dt) * 180.0 / math.pi / self.dt

        # Penalize excessive turn rates
        max_turn_tensor = torch.tensor(self.max_turn_rate, device=velocities.device)
        excess_turn = F.relu(turn_rate - max_turn_tensor)

        residual = torch.mean(excess_turn ** 2)

        return residual

    def speed_constraint(self, velocities: torch.Tensor) -> torch.Tensor:
        """
        Speed constraint: limit speed range

        Args:
            velocities: Velocity tensor (batch_size, seq_len, 2)

        Returns:
            Speed constraint residual
        """
        # Compute speed magnitude
        speeds = torch.norm(velocities, dim=-1)  # (batch_size, seq_len)

        # Convert from nm/min to knots (1 nm/min = 60 knots)
        speeds_knots = speeds * 60.0

        # Penalize speeds outside valid range
        min_speed_tensor = torch.tensor(self.min_speed, device=velocities.device)
        max_speed_tensor = torch.tensor(self.max_speed, device=velocities.device)

        speed_violation_low = F.relu(min_speed_tensor - speeds_knots)
        speed_violation_high = F.relu(speeds_knots - max_speed_tensor)

        residual = torch.mean(speed_violation_low ** 2 + speed_violation_high ** 2)

        return residual

    def continuity_constraint(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Continuity constraint: trajectory should be smooth

        Args:
            positions: Position tensor (batch_size, seq_len, 2)

        Returns:
            Continuity constraint residual
        """
        if positions.size(1) < 3:
            return torch.tensor(0.0, device=positions.device)

        # Second derivative (discrete approximation of curvature)
        d2p_dt2 = (positions[:, 2:, :] - 2 * positions[:, 1:-1, :] +
                   positions[:, :-2, :]) / (self.dt ** 2)

        # Penalize high curvature (non-smooth trajectories)
        curvature = torch.norm(d2p_dt2, dim=-1)  # (batch_size, seq_len-2)

        residual = torch.mean(curvature ** 2)

        return residual


class TrajPINN(nn.Module):
    """Physics-Informed Neural Network for ship trajectory prediction."""

    def __init__(self, config: Dict):
        """
        Initialize PINN model.

        Args:
            config: Model configuration dictionary
        """
        super(TrajPINN, self).__init__()

        self.config = config
        arch_config = config['model']['architecture']

        # Architecture parameters
        self.d_input = arch_config['d_input']
        self.hidden_dims = arch_config['hidden_dims']
        self.dropout = arch_config['dropout']

        # Output parameters
        self.forecast_steps = arch_config['forecast_steps']
        self.output_dim = arch_config['output_dim']

        # Physics parameters
        self.dt = arch_config.get('dt', 1.0)  # Time step in minutes
        self.use_physics = arch_config.get('use_physics', True)

        # Build model
        self._build_model()

        # Physics module
        if self.use_physics:
            self.physics = PhysicsModule(self.dt)

    def _build_model(self):
        """Build PINN architecture."""
        # Input normalization
        self.input_norm = nn.LayerNorm(self.d_input)

        # Feature extraction network
        layers = []
        prev_dim = self.d_input

        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.feature_network = nn.Sequential(*layers)

        # Temporal modeling
        self.temporal_lstm = nn.LSTM(
            input_size=prev_dim,
            hidden_size=prev_dim,
            num_layers=2,
            dropout=self.dropout if len(self.hidden_dims) > 1 else 0,
            batch_first=True
        )

        # Position prediction head
        self.position_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(prev_dim // 2, self.forecast_steps * self.output_dim)
        )

        # Velocity prediction head (for physics constraints)
        self.velocity_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(prev_dim // 2, self.forecast_steps * self.output_dim)
        )

        # Physics-aware fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(prev_dim * 2, prev_dim),
            nn.Tanh(),
            nn.Linear(prev_dim, prev_dim)
        )

        # Physics feature projection (for dimensional alignment)
        self.physics_proj = nn.Linear(2, prev_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x: torch.Tensor,
                return_physics_terms: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, d_input)
            return_physics_terms: Whether to return physics constraint terms

        Returns:
            Predicted trajectories (batch_size, forecast_steps, output_dim)
            If return_physics_terms=True, also returns physics terms
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Input normalization
        x_norm = self.input_norm(x)

        # Feature extraction
        features = self.feature_network(x_norm)  # (batch_size, seq_len, hidden_dim)

        # Temporal modeling
        temporal_features, (hidden, cell) = self.temporal_lstm(features)

        # Use last time step representation
        final_features = temporal_features[:, -1, :]  # (batch_size, hidden_dim)

        # Predict positions and velocities
        position_predictions = self.position_head(final_features)
        position_predictions = position_predictions.view(batch_size, self.forecast_steps, self.output_dim)

        velocity_predictions = self.velocity_head(final_features)
        velocity_predictions = velocity_predictions.view(batch_size, self.forecast_steps, self.output_dim)

        # Physics-informed integration
        if self.use_physics and self.training:
            # Use physics constraints to refine predictions
            physics_features = self._apply_physics_constraints(
                position_predictions, velocity_predictions, x
            )

            # Fusion of data-driven and physics-informed predictions
            combined_features = torch.cat([final_features, physics_features], dim=-1)
            fused_features = self.fusion_layer(combined_features)

            # Refined position predictions
            refined_positions = self.position_head(fused_features)
            refined_positions = refined_positions.view(batch_size, self.forecast_steps, self.output_dim)

            predictions = refined_positions
        else:
            predictions = position_predictions

        if return_physics_terms and self.use_physics:
            physics_terms = self._compute_physics_terms(predictions, velocity_predictions, x)
            return predictions, physics_terms

        return predictions

    def _apply_physics_constraints(self, positions: torch.Tensor,
                                 velocities: torch.Tensor,
                                 input_history: torch.Tensor) -> torch.Tensor:
        """Apply physics constraints to generate physics-informed features."""
        batch_size = positions.shape[0]
        device = positions.device

        # Get last known position and velocity from input
        last_position = input_history[:, -1, :2]  # (batch_size, 2)
        last_velocity = input_history[:, -1, 2:4] if input_history.size(-1) > 2 else torch.zeros_like(last_position)

        # Initialize physics-based trajectory
        physics_positions = []
        current_pos = last_position
        current_vel = last_velocity

        for t in range(self.forecast_steps):
            # Physics-based position update: p(t+1) = p(t) + v(t) * dt
            next_pos = current_pos + current_vel * self.dt
            physics_positions.append(next_pos)

            # Update velocity (simple integration with predicted velocity)
            if t < velocities.size(1):
                # Blend predicted velocity with physics-based velocity
                predicted_vel = velocities[:, t, :]
                alpha = 0.7  # Weight for predicted velocity
                current_vel = alpha * predicted_vel + (1 - alpha) * current_vel

            current_pos = next_pos

        physics_trajectory = torch.stack(physics_positions, dim=1)  # (batch_size, forecast_steps, 2)

        # Compute physics-informed features (match final_features dimension)
        physics_features = torch.mean(physics_trajectory, dim=1)  # Global representation (batch_size, 2)

        # Project to correct dimension
        physics_features = self.physics_proj(physics_features)

        return physics_features

    def _compute_physics_terms(self, positions: torch.Tensor,
                             velocities: torch.Tensor,
                             input_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute physics constraint terms for loss calculation."""
        physics_terms = {}

        # Combine input history positions with predictions
        input_positions = input_history[:, :, :2]  # (batch_size, seq_len, 2)
        full_positions = torch.cat([input_positions, positions], dim=1)

        # Combine input history velocities with predictions
        if input_history.size(-1) > 2:
            input_velocities = input_history[:, :, 2:4]  # (batch_size, seq_len, 2)
            full_velocities = torch.cat([input_velocities, velocities], dim=1)
        else:
            # Compute velocities from positions if not available
            full_velocities = torch.zeros_like(full_positions)
            full_velocities[:, 1:, :] = (full_positions[:, 1:, :] - full_positions[:, :-1, :]) / self.dt

        # Compute physics constraints
        physics_terms['kinematic'] = self.physics.kinematic_constraint(full_positions, full_velocities)
        physics_terms['acceleration'] = self.physics.acceleration_constraint(full_velocities)
        physics_terms['turning'] = self.physics.turning_constraint(full_velocities)
        physics_terms['speed'] = self.physics.speed_constraint(full_velocities)
        physics_terms['continuity'] = self.physics.continuity_constraint(full_positions)

        return physics_terms

    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'TrajPINN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': {
                'input_dim': self.d_input,
                'hidden_dims': self.hidden_dims,
                'forecast_steps': self.forecast_steps,
                'output_dim': self.output_dim,
                'use_physics': self.use_physics,
                'dt': self.dt
            }
        }


def create_pinn_model(config: Dict, model_type: str = 'basic') -> nn.Module:
    """
    Factory function to create PINN models.

    Args:
        config: Model configuration
        model_type: Type of PINN model

    Returns:
        PINN model instance
    """
    if model_type == 'basic':
        return TrajPINN(config)
    else:
        raise ValueError(f"Unknown PINN model type: {model_type}")


class PINNLoss(nn.Module):
    """Custom loss function for PINN trajectory prediction with physics constraints."""

    def __init__(self, loss_type: str = 'mse',
                 physics_weights: Optional[Dict[str, float]] = None):
        """
        Initialize PINN loss.

        Args:
            loss_type: Base loss type
            physics_weights: Weights for different physics constraints
        """
        super(PINNLoss, self).__init__()

        self.loss_type = loss_type

        # Default physics weights
        self.physics_weights = physics_weights or {
            'kinematic': 1.0,
            'acceleration': 0.5,
            'turning': 0.3,
            'speed': 0.2,
            'continuity': 0.4
        }

        # Base loss function
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'smooth_l1':
            self.base_loss = nn.SmoothL1Loss()
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                physics_terms: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate total PINN loss.

        Args:
            predictions: Predicted trajectories
            targets: Target trajectories
            physics_terms: Physics constraint terms

        Returns:
            Total loss and loss components
        """
        # Data loss (MSE between predictions and targets)
        data_loss = self.base_loss(predictions, targets)

        loss_components = {'data_loss': data_loss}
        total_loss = data_loss

        # Physics constraints
        if physics_terms is not None:
            physics_loss = torch.tensor(0.0, device=predictions.device)

            for constraint_name, constraint_value in physics_terms.items():
                if constraint_name in self.physics_weights:
                    weighted_constraint = self.physics_weights[constraint_name] * constraint_value
                    physics_loss += weighted_constraint
                    loss_components[f'physics_{constraint_name}'] = weighted_constraint

            loss_components['physics_loss'] = physics_loss
            total_loss += physics_loss

        loss_components['total_loss'] = total_loss

        return total_loss, loss_components

    def update_physics_weights(self, new_weights: Dict[str, float]):
        """Update physics constraint weights."""
        self.physics_weights.update(new_weights)


# Adaptive physics weight scheduler
class PhysicsWeightScheduler:
    """Scheduler for physics constraint weights during training."""

    def __init__(self, initial_weights: Dict[str, float],
                 schedule_type: str = 'linear',
                 warmup_epochs: int = 10,
                 max_epochs: int = 100):
        """
        Initialize physics weight scheduler.

        Args:
            initial_weights: Initial physics weights
            schedule_type: Type of scheduling ('linear', 'exponential', 'cosine')
            warmup_epochs: Number of epochs to warm up physics constraints
            max_epochs: Maximum training epochs
        """
        self.initial_weights = initial_weights.copy()
        self.current_weights = initial_weights.copy()
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

    def step(self, epoch: int):
        """Update physics weights based on current epoch."""
        if epoch < self.warmup_epochs:
            # Gradually increase physics weights during warmup
            alpha = epoch / self.warmup_epochs
            for key in self.current_weights:
                self.current_weights[key] = alpha * self.initial_weights[key]
        else:
            # Apply scheduling after warmup
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)

            if self.schedule_type == 'linear':
                multiplier = 1.0 + progress
            elif self.schedule_type == 'exponential':
                multiplier = math.exp(progress)
            elif self.schedule_type == 'cosine':
                multiplier = 1.0 + 0.5 * (1 + math.cos(math.pi * progress))
            else:
                multiplier = 1.0

            for key in self.current_weights:
                self.current_weights[key] = multiplier * self.initial_weights[key]

        return self.current_weights


if __name__ == "__main__":
    # Example usage
    config = {
        'model': {
            'architecture': {
                'd_input': 5,
                'hidden_dims': [256, 256, 128],
                'dropout': 0.1,
                'forecast_steps': 30,
                'output_dim': 2,
                'dt': 1.0,
                'use_physics': True
            }
        }
    }

    # Create model
    model = create_pinn_model(config)
    print(f"Model info: {model.get_model_info()}")

    # Test forward pass
    batch_size, seq_len = 16, 60
    x = torch.randn(batch_size, seq_len, 5)
    y = torch.randn(batch_size, 30, 2)

    model.train()
    predictions, physics_terms = model(x, return_physics_terms=True)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Physics terms: {list(physics_terms.keys())}")

    # Test loss
    loss_fn = PINNLoss('mse')
    total_loss, loss_components = loss_fn(predictions, y, physics_terms)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss components: {[f'{k}: {v.item():.4f}' for k, v in loss_components.items()]}")

    # Test physics weight scheduler
    scheduler = PhysicsWeightScheduler(
        initial_weights={'kinematic': 1.0, 'acceleration': 0.5},
        schedule_type='linear',
        warmup_epochs=5,
        max_epochs=50
    )

    for epoch in range(10):
        weights = scheduler.step(epoch)
        print(f"Epoch {epoch}: {weights}")