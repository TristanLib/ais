"""
Transformer model for ship trajectory prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate div_term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        if x.dim() == 3 and x.size(0) != self.pe.size(0):
            # Input is (batch_size, seq_len, d_model)
            seq_len = x.size(1)
            x = x + self.pe[:seq_len, :, :].transpose(0, 1)
        else:
            # Input is (seq_len, batch_size, d_model)
            x = x + self.pe[:x.size(0), :]

        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        """Initialize learnable positional encoding."""
        super(LearnablePositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learnable positional encoding."""
        if x.dim() == 3 and x.size(1) <= self.pe.size(0):
            # Input is (batch_size, seq_len, d_model)
            seq_len = x.size(1)
            x = x + self.pe[:seq_len, :].unsqueeze(0)

        return self.dropout(x)


class TrajTransformer(nn.Module):
    """Transformer-based trajectory prediction model."""

    def __init__(self, config: Dict):
        """
        Initialize Transformer model.

        Args:
            config: Model configuration dictionary
        """
        super(TrajTransformer, self).__init__()

        self.config = config
        arch_config = config['model']['architecture']

        # Architecture parameters
        self.d_input = arch_config['d_input']
        self.d_model = arch_config['d_model']
        self.n_heads = arch_config['n_heads']
        self.n_layers = arch_config['n_layers']
        self.d_ff = arch_config['d_ff']
        self.dropout = arch_config['dropout']

        # Output parameters
        self.forecast_steps = arch_config['forecast_steps']
        self.output_dim = arch_config['output_dim']

        # Positional encoding type
        self.pos_encoding_type = arch_config.get('pos_encoding_type', 'sinusoidal')
        self.max_seq_len = arch_config.get('max_seq_len', 512)

        # Neighbor attention
        self.use_neighbor_attention = arch_config.get('use_neighbor_attention', False)
        self.neighbor_d_model = arch_config.get('neighbor_d_model', 64)

        # Build model
        self._build_model()

    def _build_model(self):
        """Build transformer architecture."""
        # Input embedding
        self.input_embedding = nn.Linear(self.d_input, self.d_model)

        # Positional encoding
        if self.pos_encoding_type == 'sinusoidal':
            self.pos_encoding = PositionalEncoding(
                self.d_model, self.max_seq_len, self.dropout
            )
        elif self.pos_encoding_type == 'learned':
            self.pos_encoding = LearnablePositionalEncoding(
                self.d_model, self.max_seq_len, self.dropout
            )
        else:
            self.pos_encoding = None

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers
        )

        # Neighbor attention (optional)
        if self.use_neighbor_attention:
            self.neighbor_attention = nn.MultiheadAttention(
                embed_dim=self.neighbor_d_model,
                num_heads=4,
                dropout=self.dropout,
                batch_first=True
            )
            self.neighbor_proj = nn.Linear(self.d_input, self.neighbor_d_model)
            self.neighbor_fusion = nn.Linear(
                self.d_model + self.neighbor_d_model, self.d_model
            )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.forecast_steps * self.output_dim)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)

    def forward(self, x: torch.Tensor, neighbors: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, d_input)
            neighbors: Neighbor features (batch_size, n_neighbors, d_input)
            mask: Attention mask (seq_len, seq_len)

        Returns:
            Predicted trajectories (batch_size, forecast_steps, output_dim)
        """
        batch_size = x.size(0)

        # Input embedding
        embedded = self.input_embedding(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        if self.pos_encoding is not None:
            embedded = self.pos_encoding(embedded)

        # Transformer encoding
        encoded = self.transformer_encoder(embedded, mask=mask)  # (batch_size, seq_len, d_model)

        # Global representation (use last token or mean pooling)
        global_repr = encoded[:, -1, :]  # (batch_size, d_model)

        # Neighbor attention (optional)
        if self.use_neighbor_attention and neighbors is not None:
            neighbor_repr = self._process_neighbors(x, neighbors)
            global_repr = self.neighbor_fusion(
                torch.cat([global_repr, neighbor_repr], dim=-1)
            )

        # Generate predictions
        predictions = self.output_head(global_repr)  # (batch_size, forecast_steps * output_dim)
        predictions = predictions.view(batch_size, self.forecast_steps, self.output_dim)

        return predictions

    def _process_neighbors(self, target: torch.Tensor,
                          neighbors: torch.Tensor) -> torch.Tensor:
        """
        Process neighbor information using attention.

        Args:
            target: Target vessel features (batch_size, seq_len, d_input)
            neighbors: Neighbor features (batch_size, n_neighbors, d_input)

        Returns:
            Neighbor representation (batch_size, neighbor_d_model)
        """
        # Project to neighbor space
        target_proj = self.neighbor_proj(target[:, -1:, :])  # (batch_size, 1, neighbor_d_model)
        neighbor_proj = self.neighbor_proj(neighbors)  # (batch_size, n_neighbors, neighbor_d_model)

        # Cross-attention: target attends to neighbors
        neighbor_repr, _ = self.neighbor_attention(
            target_proj, neighbor_proj, neighbor_proj
        )  # (batch_size, 1, neighbor_d_model)

        return neighbor_repr.squeeze(1)  # (batch_size, neighbor_d_model)

    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'TrajTransformer',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': {
                'input_dim': self.d_input,
                'model_dim': self.d_model,
                'num_heads': self.n_heads,
                'num_layers': self.n_layers,
                'ff_dim': self.d_ff,
                'forecast_steps': self.forecast_steps,
                'output_dim': self.output_dim,
                'pos_encoding': self.pos_encoding_type,
                'neighbor_attention': self.use_neighbor_attention
            }
        }


class TransformerDecoderModel(nn.Module):
    """Transformer encoder-decoder model for trajectory prediction."""

    def __init__(self, config: Dict):
        """Initialize encoder-decoder transformer."""
        super(TransformerDecoderModel, self).__init__()

        self.config = config
        arch_config = config['model']['architecture']

        # Architecture parameters
        self.d_input = arch_config['d_input']
        self.d_model = arch_config['d_model']
        self.n_heads = arch_config['n_heads']
        self.n_layers = arch_config['n_layers']
        self.d_ff = arch_config['d_ff']
        self.dropout = arch_config['dropout']

        # Output parameters
        self.forecast_steps = arch_config['forecast_steps']
        self.output_dim = arch_config['output_dim']

        self._build_model()

    def _build_model(self):
        """Build encoder-decoder architecture."""
        # Input embeddings
        self.input_embedding = nn.Linear(self.d_input, self.d_model)
        self.output_embedding = nn.Linear(self.output_dim, self.d_model)

        # Positional encodings
        self.pos_encoding = PositionalEncoding(self.d_model, dropout=self.dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.n_heads,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=self.n_layers,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.output_dim)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with encoder-decoder architecture.

        Args:
            x: Input sequence (batch_size, seq_len, d_input)
            target: Target sequence for teacher forcing (batch_size, forecast_steps, output_dim)

        Returns:
            Predicted trajectories (batch_size, forecast_steps, output_dim)
        """
        batch_size = x.size(0)
        device = x.device

        # Encode input sequence
        src_embedded = self.input_embedding(x)
        src_embedded = self.pos_encoding(src_embedded)

        # Prepare decoder input
        if target is not None and self.training:
            # Teacher forcing: use ground truth
            tgt_embedded = self.output_embedding(target)
        else:
            # Inference: start with last position
            start_pos = x[:, -1, :self.output_dim].unsqueeze(1)  # (batch_size, 1, output_dim)
            tgt_embedded = self.output_embedding(start_pos)

        tgt_embedded = self.pos_encoding(tgt_embedded)

        # Create causal mask for decoder
        tgt_len = tgt_embedded.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(device)

        # Transformer forward pass
        if target is not None and self.training:
            # Training with teacher forcing
            output = self.transformer(src_embedded, tgt_embedded, tgt_mask=tgt_mask)
            predictions = self.output_proj(output)
        else:
            # Autoregressive inference
            predictions = self._generate_autoregressive(src_embedded, tgt_embedded, device)

        return predictions

    def _generate_autoregressive(self, src_embedded: torch.Tensor,
                               start_embedded: torch.Tensor,
                               device: torch.device) -> torch.Tensor:
        """Generate predictions autoregressively."""
        batch_size = src_embedded.size(0)
        predictions = []

        tgt_embedded = start_embedded  # (batch_size, 1, d_model)

        for _ in range(self.forecast_steps):
            tgt_len = tgt_embedded.size(1)
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(device)

            # Forward pass
            output = self.transformer(src_embedded, tgt_embedded, tgt_mask=tgt_mask)

            # Get last prediction
            last_output = output[:, -1:, :]  # (batch_size, 1, d_model)
            pred_step = self.output_proj(last_output)  # (batch_size, 1, output_dim)

            predictions.append(pred_step)

            # Update decoder input
            next_embedded = self.output_embedding(pred_step)
            next_embedded = self.pos_encoding(next_embedded)
            tgt_embedded = torch.cat([tgt_embedded, next_embedded], dim=1)

        # Concatenate all predictions
        predictions = torch.cat(predictions, dim=1)  # (batch_size, forecast_steps, output_dim)

        return predictions


class TransformerLoss(nn.Module):
    """Custom loss function for transformer trajectory prediction."""

    def __init__(self, loss_type: str = 'smooth_l1', step_weights: str = 'linear',
                 coord_smoothness_weight: float = 0.01,
                 direction_consistency_weight: float = 0.005):
        """
        Initialize transformer loss.

        Args:
            loss_type: Base loss type
            step_weights: Step weighting scheme
            coord_smoothness_weight: Weight for coordinate smoothness regularization
            direction_consistency_weight: Weight for direction consistency regularization
        """
        super(TransformerLoss, self).__init__()

        self.loss_type = loss_type
        self.step_weights = step_weights
        self.coord_smoothness_weight = coord_smoothness_weight
        self.direction_consistency_weight = direction_consistency_weight

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
        """Calculate total loss with regularization."""
        # Base prediction loss
        pred_loss = self._calculate_prediction_loss(predictions, targets)

        # Regularization terms
        smoothness_loss = self._calculate_smoothness_loss(predictions)
        direction_loss = self._calculate_direction_consistency_loss(predictions)

        # Total loss
        total_loss = (pred_loss +
                     self.coord_smoothness_weight * smoothness_loss +
                     self.direction_consistency_weight * direction_loss)

        return total_loss

    def _calculate_prediction_loss(self, predictions: torch.Tensor,
                                 targets: torch.Tensor) -> torch.Tensor:
        """Calculate weighted prediction loss."""
        # Base loss
        loss = self.base_loss(predictions, targets)  # (batch_size, forecast_steps, output_dim)
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

        return (loss * weights).mean()

    def _calculate_smoothness_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Calculate coordinate smoothness regularization."""
        if predictions.size(1) < 2:
            return torch.tensor(0.0, device=predictions.device)

        # Calculate differences between consecutive predictions
        diffs = predictions[:, 1:, :] - predictions[:, :-1, :]  # (batch_size, T-1, output_dim)
        smoothness_loss = torch.mean(diffs ** 2)

        return smoothness_loss

    def _calculate_direction_consistency_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Calculate direction consistency regularization."""
        if predictions.size(1) < 3:
            return torch.tensor(0.0, device=predictions.device)

        # Calculate direction vectors
        directions1 = predictions[:, 1:-1, :] - predictions[:, :-2, :]  # (batch_size, T-2, output_dim)
        directions2 = predictions[:, 2:, :] - predictions[:, 1:-1, :]   # (batch_size, T-2, output_dim)

        # Normalize directions
        directions1_norm = F.normalize(directions1, p=2, dim=-1)
        directions2_norm = F.normalize(directions2, p=2, dim=-1)

        # Calculate cosine similarity (want it to be close to 1)
        cosine_sim = torch.sum(directions1_norm * directions2_norm, dim=-1)  # (batch_size, T-2)
        direction_loss = torch.mean(1 - cosine_sim)

        return direction_loss


def create_transformer_model(config: Dict, model_type: str = 'encoder') -> nn.Module:
    """
    Factory function to create transformer models.

    Args:
        config: Model configuration
        model_type: Type of transformer ('encoder' or 'encoder_decoder')

    Returns:
        Transformer model instance
    """
    if model_type == 'encoder':
        return TrajTransformer(config)
    elif model_type == 'encoder_decoder':
        return TransformerDecoderModel(config)
    else:
        raise ValueError(f"Unknown transformer type: {model_type}")


if __name__ == "__main__":
    # Example usage
    config = {
        'model': {
            'architecture': {
                'd_input': 5,
                'd_model': 128,
                'n_heads': 8,
                'n_layers': 6,
                'd_ff': 512,
                'dropout': 0.1,
                'forecast_steps': 30,
                'output_dim': 2,
                'pos_encoding_type': 'sinusoidal',
                'max_seq_len': 512,
                'use_neighbor_attention': False
            }
        }
    }

    # Create model
    model = create_transformer_model(config, 'encoder')
    print(f"Model info: {model.get_model_info()}")

    # Test forward pass
    batch_size, seq_len = 16, 60
    x = torch.randn(batch_size, seq_len, 5)
    y = torch.randn(batch_size, 30, 2)

    model.eval()
    with torch.no_grad():
        predictions = model(x)
        print(f"Predictions shape: {predictions.shape}")

    # Test loss
    loss_fn = TransformerLoss('smooth_l1', 'linear')
    loss = loss_fn(predictions, y)
    print(f"Loss: {loss.item():.4f}")
