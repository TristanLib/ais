"""
LSTM model for ship trajectory prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class TrajLSTM(nn.Module):
    """LSTM-based trajectory prediction model."""

    def __init__(self, config: Dict):
        """
        Initialize LSTM model.

        Args:
            config: Model configuration dictionary
        """
        super(TrajLSTM, self).__init__()

        self.config = config
        arch_config = config['model']['architecture']

        # Architecture parameters
        self.d_input = arch_config['d_input']
        self.hidden_size = arch_config['hidden_size']
        self.n_layers = arch_config['n_layers']
        self.dropout = arch_config['dropout']
        self.bidirectional = arch_config.get('bidirectional', False)

        # Output parameters
        self.forecast_steps = arch_config['forecast_steps']
        self.output_dim = arch_config['output_dim']

        # MLP head parameters
        self.head_hidden = arch_config.get('head_hidden', 128)
        self.head_dropout = arch_config.get('head_dropout', 0.1)

        # Build model
        self._build_model()

        # Teacher forcing for training
        self.teacher_forcing_ratio = config['model'].get('teacher_forcing_ratio', 0.5)

    def _build_model(self):
        """Build LSTM architecture."""
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.d_input,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout if self.n_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )

        # Calculate LSTM output size
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(lstm_output_size, self.head_hidden),
            nn.ReLU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.head_hidden, self.output_dim * self.forecast_steps)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_input)
            target: Target tensor for teacher forcing (batch_size, forecast_steps, output_dim)

        Returns:
            Predicted trajectories of shape (batch_size, forecast_steps, output_dim)
        """
        batch_size = x.size(0)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch_size, seq_len, hidden_size)

        # Use last output for prediction
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Direct prediction approach
        predictions = self.head(last_output)  # (batch_size, forecast_steps * output_dim)
        predictions = predictions.view(batch_size, self.forecast_steps, self.output_dim)

        return predictions

    def forward_sequence_to_sequence(self, x: torch.Tensor,
                                   target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sequence-to-sequence forward pass with optional teacher forcing.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_input)
            target: Target tensor for teacher forcing (batch_size, forecast_steps, output_dim)

        Returns:
            Predicted trajectories of shape (batch_size, forecast_steps, output_dim)
        """
        batch_size = x.size(0)
        device = x.device

        # Encode input sequence
        encoder_out, (hidden, cell) = self.lstm(x)

        # Initialize decoder
        predictions = []
        decoder_input = x[:, -1:, :self.output_dim]  # Use last position as initial input

        # Decode step by step
        for t in range(self.forecast_steps):
            # Decoder step
            decoder_out, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))

            # Predict next position
            pred_step = self.head(decoder_out.squeeze(1))  # (batch_size, output_dim)
            pred_step = pred_step.view(batch_size, 1, self.output_dim)
            predictions.append(pred_step)

            # Teacher forcing decision
            if self.training and target is not None and np.random.random() < self.teacher_forcing_ratio:
                # Use ground truth
                decoder_input = target[:, t:t+1, :]
            else:
                # Use prediction
                # Prepare next input (need to reconstruct full feature vector)
                next_input = torch.zeros(batch_size, 1, self.d_input, device=device)
                next_input[:, :, :self.output_dim] = pred_step

                # Copy other features from last input (SOG, COG, etc.)
                if self.d_input > self.output_dim:
                    next_input[:, :, self.output_dim:] = decoder_input[:, :, self.output_dim:]

                decoder_input = next_input

        # Concatenate predictions
        predictions = torch.cat(predictions, dim=1)  # (batch_size, forecast_steps, output_dim)

        return predictions

    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'TrajLSTM',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': {
                'input_dim': self.d_input,
                'hidden_size': self.hidden_size,
                'num_layers': self.n_layers,
                'bidirectional': self.bidirectional,
                'forecast_steps': self.forecast_steps,
                'output_dim': self.output_dim
            }
        }


class LSTMWithAttention(nn.Module):
    """LSTM with attention mechanism for trajectory prediction."""

    def __init__(self, config: Dict):
        """Initialize LSTM with attention."""
        super(LSTMWithAttention, self).__init__()

        self.config = config
        arch_config = config['model']['architecture']

        # Base LSTM
        self.d_input = arch_config['d_input']
        self.hidden_size = arch_config['hidden_size']
        self.n_layers = arch_config['n_layers']
        self.dropout = arch_config['dropout']

        # Output parameters
        self.forecast_steps = arch_config['forecast_steps']
        self.output_dim = arch_config['output_dim']

        # Build model
        self._build_model()

    def _build_model(self):
        """Build LSTM with attention architecture."""
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=self.d_input,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout if self.n_layers > 1 else 0,
            batch_first=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(self.hidden_size, self.output_dim * self.forecast_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_input)

        Returns:
            Predicted trajectories of shape (batch_size, forecast_steps, output_dim)
        """
        batch_size = x.size(0)

        # Encode sequence
        encoder_out, _ = self.encoder(x)  # (batch_size, seq_len, hidden_size)

        # Self-attention over encoded sequence
        attended_out, attention_weights = self.attention(
            encoder_out, encoder_out, encoder_out
        )  # (batch_size, seq_len, hidden_size)

        # Global representation (weighted average)
        global_repr = attended_out.mean(dim=1)  # (batch_size, hidden_size)

        # Predict future trajectory
        predictions = self.output_proj(global_repr)  # (batch_size, forecast_steps * output_dim)
        predictions = predictions.view(batch_size, self.forecast_steps, self.output_dim)

        return predictions


def create_lstm_model(config: Dict, model_type: str = 'basic') -> nn.Module:
    """
    Factory function to create LSTM models.

    Args:
        config: Model configuration
        model_type: Type of LSTM model ('basic' or 'attention')

    Returns:
        LSTM model instance
    """
    if model_type == 'basic':
        return TrajLSTM(config)
    elif model_type == 'attention':
        return LSTMWithAttention(config)
    else:
        raise ValueError(f"Unknown LSTM model type: {model_type}")


class LSTMLoss(nn.Module):
    """Custom loss function for LSTM trajectory prediction."""

    def __init__(self, loss_type: str = 'mse', step_weights: str = 'uniform'):
        """
        Initialize loss function.

        Args:
            loss_type: Type of loss ('mse', 'smooth_l1', 'huber')
            step_weights: Step weighting scheme ('uniform', 'linear', 'exponential')
        """
        super(LSTMLoss, self).__init__()

        self.loss_type = loss_type
        self.step_weights = step_weights

        # Base loss function
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.base_loss = nn.SmoothL1Loss(reduction='none')
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss.

        Args:
            predictions: Predicted trajectories (batch_size, forecast_steps, output_dim)
            targets: Target trajectories (batch_size, forecast_steps, output_dim)

        Returns:
            Scalar loss value
        """
        # Calculate base loss
        loss = self.base_loss(predictions, targets)  # (batch_size, forecast_steps, output_dim)

        # Average over output dimensions
        loss = loss.mean(dim=-1)  # (batch_size, forecast_steps)

        # Apply step weights
        if self.step_weights == 'uniform':
            weights = torch.ones_like(loss)
        elif self.step_weights == 'linear':
            steps = torch.arange(1, loss.size(1) + 1, device=loss.device, dtype=torch.float)
            weights = steps / steps.sum()
            weights = weights.unsqueeze(0).expand_as(loss)
        elif self.step_weights == 'exponential':
            steps = torch.arange(loss.size(1), device=loss.device, dtype=torch.float)
            weights = torch.exp(0.1 * steps)
            weights = weights / weights.sum()
            weights = weights.unsqueeze(0).expand_as(loss)
        else:
            weights = torch.ones_like(loss)

        # Weighted loss
        weighted_loss = loss * weights

        # Average over batch and time steps
        return weighted_loss.mean()


if __name__ == "__main__":
    # Example usage
    import yaml

    # Sample configuration
    config = {
        'model': {
            'architecture': {
                'd_input': 5,
                'hidden_size': 128,
                'n_layers': 2,
                'dropout': 0.2,
                'bidirectional': False,
                'forecast_steps': 30,
                'output_dim': 2,
                'head_hidden': 64,
                'head_dropout': 0.1
            },
            'teacher_forcing_ratio': 0.5
        }
    }

    # Create model
    model = create_lstm_model(config, 'basic')
    print(f"Model info: {model.get_model_info()}")

    # Test forward pass
    batch_size, seq_len = 16, 60
    x = torch.randn(batch_size, seq_len, 5)
    y = torch.randn(batch_size, 30, 2)

    model.train()
    predictions = model(x, y)
    print(f"Predictions shape: {predictions.shape}")

    # Test loss
    loss_fn = LSTMLoss('smooth_l1', 'linear')
    loss = loss_fn(predictions, y)
    print(f"Loss: {loss.item():.4f}")
