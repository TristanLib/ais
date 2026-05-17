"""
Spatio-Temporal Transformer (STT) model for ship trajectory prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SpatialPositionalEncoding(nn.Module):
    """Spatial positional encoding for geographic coordinates."""

    def __init__(self, d_model: int, max_distance: float = 100.0):
        """
        Initialize spatial positional encoding.

        Args:
            d_model: Model dimension
            max_distance: Maximum distance for normalization (nautical miles)
        """
        super(SpatialPositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_distance = max_distance

        # Learnable spatial embedding
        self.spatial_embedding = nn.Linear(2, d_model)  # lat, lon -> d_model

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode spatial positions.

        Args:
            positions: Position tensor (batch_size, seq_len, 2) or (batch_size, n_entities, 2)

        Returns:
            Spatial encoding (batch_size, seq_len, d_model) or (batch_size, n_entities, d_model)
        """
        # Normalize positions
        normalized_pos = positions / self.max_distance

        # Apply spatial embedding
        spatial_enc = self.spatial_embedding(normalized_pos)

        return spatial_enc


class TemporalAttention(nn.Module):
    """Temporal attention mechanism."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        """
        Initialize temporal attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(TemporalAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for temporal attention.

        Args:
            query: Query tensor (batch_size, seq_len_q, d_model)
            key: Key tensor (batch_size, seq_len_k, d_model)
            value: Value tensor (batch_size, seq_len_v, d_model)
            mask: Attention mask

        Returns:
            Attended features (batch_size, seq_len_q, d_model)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]

        residual = query

        # Linear projections
        Q = self.q_linear(query).view(batch_size, seq_len_q, self.n_heads, self.head_dim)
        K = self.k_linear(key).view(batch_size, seq_len_k, self.n_heads, self.head_dim)
        V = self.v_linear(value).view(batch_size, seq_len_k, self.n_heads, self.head_dim)

        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch_size, n_heads, seq_len_q, head_dim)
        K = K.transpose(1, 2)  # (batch_size, n_heads, seq_len_k, head_dim)
        V = V.transpose(1, 2)  # (batch_size, n_heads, seq_len_v, head_dim)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (batch_size, n_heads, seq_len_q, head_dim)

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )

        # Output projection
        output = self.out_linear(attended)

        # Residual connection and layer normalization
        output = self.layer_norm(output + residual)

        return output


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for inter-ship relationships."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        """
        Initialize spatial attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(SpatialAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Distance-based attention weighting
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, features: torch.Tensor, positions: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for spatial attention.

        Args:
            features: Feature tensor (batch_size, n_entities, d_model)
            positions: Position tensor (batch_size, n_entities, 2)
            mask: Attention mask

        Returns:
            Spatially attended features (batch_size, n_entities, d_model)
        """
        batch_size, n_entities, _ = features.shape

        residual = features

        # Calculate pairwise distances
        pos_expanded = positions.unsqueeze(2)  # (batch_size, n_entities, 1, 2)
        pos_tiled = positions.unsqueeze(1)     # (batch_size, 1, n_entities, 2)

        distances = torch.norm(pos_expanded - pos_tiled, dim=-1)  # (batch_size, n_entities, n_entities)

        # Distance-based attention weights
        distance_weights = self.distance_mlp(distances.unsqueeze(-1)).squeeze(-1)  # (batch_size, n_entities, n_entities)
        distance_weights = F.softmax(distance_weights, dim=-1)

        # Self-attention
        attended, attention_weights = self.multihead_attn(
            features, features, features, attn_mask=mask
        )

        # Combine with distance weights (simplified combination)
        if mask is None:
            # Apply distance weighting to attention output
            distance_weighted = torch.bmm(distance_weights, attended)
            attended = 0.7 * attended + 0.3 * distance_weighted

        # Residual connection and normalization
        output = self.layer_norm(attended + residual)

        return output


class SpatioTemporalBlock(nn.Module):
    """Spatio-temporal transformer block."""

    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 512, dropout: float = 0.1):
        """
        Initialize spatio-temporal block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super(SpatioTemporalBlock, self).__init__()

        # Temporal attention
        self.temporal_attention = TemporalAttention(d_model, n_heads, dropout)

        # Spatial attention
        self.spatial_attention = SpatialAttention(d_model, n_heads, dropout)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.layer_norm_ff = nn.LayerNorm(d_model)

        # Fusion mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                temporal_mask: Optional[torch.Tensor] = None,
                spatial_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for spatio-temporal block.

        Args:
            x: Input tensor (batch_size, seq_len, n_entities, d_model)
            positions: Position tensor (batch_size, seq_len, n_entities, 2)
            temporal_mask: Temporal attention mask
            spatial_mask: Spatial attention mask

        Returns:
            Output tensor (batch_size, seq_len, n_entities, d_model)
        """
        batch_size, seq_len, n_entities, d_model = x.shape

        # Temporal attention: attend over time for each entity
        temporal_features = []
        for e in range(n_entities):
            entity_features = x[:, :, e, :]  # (batch_size, seq_len, d_model)
            temporal_attended = self.temporal_attention(
                entity_features, entity_features, entity_features, temporal_mask
            )
            temporal_features.append(temporal_attended.unsqueeze(2))

        temporal_output = torch.cat(temporal_features, dim=2)  # (batch_size, seq_len, n_entities, d_model)

        # Spatial attention: attend over entities for each time step
        spatial_features = []
        for t in range(seq_len):
            time_features = x[:, t, :, :]  # (batch_size, n_entities, d_model)
            time_positions = positions[:, t, :, :]  # (batch_size, n_entities, 2)

            spatial_attended = self.spatial_attention(
                time_features, time_positions, spatial_mask
            )
            spatial_features.append(spatial_attended.unsqueeze(1))

        spatial_output = torch.cat(spatial_features, dim=1)  # (batch_size, seq_len, n_entities, d_model)

        # Fusion of temporal and spatial features
        combined_features = torch.cat([temporal_output, spatial_output], dim=-1)  # (batch_size, seq_len, n_entities, 2*d_model)

        fusion_gate = self.fusion_gate(combined_features)  # (batch_size, seq_len, n_entities, d_model)
        fused_output = fusion_gate * temporal_output + (1 - fusion_gate) * spatial_output

        # Feed-forward network
        residual = fused_output
        ff_output = self.feed_forward(fused_output)
        output = self.layer_norm_ff(ff_output + residual)

        return output


class TrajSTT(nn.Module):
    """Spatio-Temporal Transformer for ship trajectory prediction."""

    def __init__(self, config: Dict):
        """
        Initialize STT model.

        Args:
            config: Model configuration dictionary
        """
        super(TrajSTT, self).__init__()

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

        # Spatial parameters
        self.max_distance = arch_config.get('max_distance', 100.0)

        # Multi-ship support
        self.max_ships = arch_config.get('max_ships', 1)

        # Build model
        self._build_model()

    def _build_model(self):
        """Build STT architecture."""
        # Input embedding
        self.input_embedding = nn.Linear(self.d_input, self.d_model)

        # Positional encodings
        self.temporal_pos_encoding = nn.Parameter(torch.randn(512, self.d_model))
        self.spatial_pos_encoding = SpatialPositionalEncoding(self.d_model, self.max_distance)

        # Spatio-temporal blocks
        self.st_blocks = nn.ModuleList([
            SpatioTemporalBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
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

        # Initialize positional encoding
        nn.init.normal_(self.temporal_pos_encoding, 0, 0.02)

    def forward(self, x: torch.Tensor, ship_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, d_input) or (batch_size, seq_len, n_ships, d_input)
            ship_mask: Mask for valid ships (batch_size, n_ships)

        Returns:
            Predicted trajectories (batch_size, forecast_steps, output_dim) or (batch_size, n_ships, forecast_steps, output_dim)
        """
        # Handle single ship case
        if x.dim() == 3:
            x = x.unsqueeze(2)  # (batch_size, seq_len, 1, d_input)
            single_ship = True
        else:
            single_ship = False

        batch_size, seq_len, n_ships, _ = x.shape
        device = x.device

        # Input embedding
        x_embedded = self.input_embedding(x)  # (batch_size, seq_len, n_ships, d_model)

        # Extract positions for spatial encoding
        positions = x[:, :, :, :2]  # Assume first 2 dimensions are lat/lon

        # Add temporal positional encoding
        temp_pos = self.temporal_pos_encoding[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, d_model)
        temp_pos = temp_pos.expand(batch_size, seq_len, n_ships, -1)

        # Add spatial positional encoding
        spatial_pos = self.spatial_pos_encoding(positions)  # (batch_size, seq_len, n_ships, d_model)

        # Combine embeddings and positional encodings
        x_encoded = x_embedded + temp_pos + spatial_pos
        x_encoded = F.dropout(x_encoded, p=self.dropout, training=self.training)

        # Apply spatio-temporal blocks
        hidden = x_encoded
        for st_block in self.st_blocks:
            hidden = st_block(hidden, positions)

        # Global pooling for prediction
        if single_ship:
            # Single ship case
            hidden = hidden.squeeze(2)  # (batch_size, seq_len, d_model)
            global_repr = hidden[:, -1, :]  # Use last time step

            predictions = self.output_projection(global_repr)  # (batch_size, forecast_steps * output_dim)
            predictions = predictions.view(batch_size, self.forecast_steps, self.output_dim)
        else:
            # Multi-ship case
            # Use last time step for each ship
            global_repr = hidden[:, -1, :, :]  # (batch_size, n_ships, d_model)

            predictions = self.output_projection(global_repr)  # (batch_size, n_ships, forecast_steps * output_dim)
            predictions = predictions.view(batch_size, n_ships, self.forecast_steps, self.output_dim)

            # Apply ship mask if provided
            if ship_mask is not None:
                mask_expanded = ship_mask.unsqueeze(-1).unsqueeze(-1)  # (batch_size, n_ships, 1, 1)
                predictions = predictions * mask_expanded

        return predictions

    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'TrajSTT',
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
                'max_distance': self.max_distance,
                'max_ships': self.max_ships
            }
        }


def create_stt_model(config: Dict, model_type: str = 'basic') -> nn.Module:
    """
    Factory function to create STT models.

    Args:
        config: Model configuration
        model_type: Type of STT model

    Returns:
        STT model instance
    """
    if model_type == 'basic':
        return TrajSTT(config)
    else:
        raise ValueError(f"Unknown STT model type: {model_type}")


class STTLoss(nn.Module):
    """Custom loss function for STT trajectory prediction."""

    def __init__(self, loss_type: str = 'smooth_l1', step_weights: str = 'linear',
                 spatial_consistency_weight: float = 0.01,
                 temporal_smoothness_weight: float = 0.005):
        """
        Initialize STT loss.

        Args:
            loss_type: Base loss type
            step_weights: Step weighting scheme
            spatial_consistency_weight: Weight for spatial consistency regularization
            temporal_smoothness_weight: Weight for temporal smoothness regularization
        """
        super(STTLoss, self).__init__()

        self.loss_type = loss_type
        self.step_weights = step_weights
        self.spatial_consistency_weight = spatial_consistency_weight
        self.temporal_smoothness_weight = temporal_smoothness_weight

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
        """Calculate total loss with spatio-temporal regularization."""
        # Base prediction loss
        pred_loss = self._calculate_prediction_loss(predictions, targets)

        # Regularization terms
        temporal_loss = self._calculate_temporal_smoothness_loss(predictions)
        spatial_loss = self._calculate_spatial_consistency_loss(predictions)

        # Total loss
        total_loss = (pred_loss +
                     self.temporal_smoothness_weight * temporal_loss +
                     self.spatial_consistency_weight * spatial_loss)

        return total_loss

    def _calculate_prediction_loss(self, predictions: torch.Tensor,
                                 targets: torch.Tensor) -> torch.Tensor:
        """Calculate weighted prediction loss."""
        loss = self.base_loss(predictions, targets)

        # Handle different tensor shapes
        if predictions.dim() == 4:  # Multi-ship case
            loss = loss.mean(dim=(1, 3))  # Average over ships and output dimensions
        else:  # Single ship case
            loss = loss.mean(dim=-1)  # Average over output dimensions

        # Apply step weights
        if self.step_weights == 'linear':
            time_dim = -1 if predictions.dim() == 3 else -2
            steps = torch.arange(1, loss.size(time_dim) + 1, device=loss.device, dtype=torch.float)
            weights = steps / steps.sum()

            # Expand weights to match loss shape
            for _ in range(loss.dim() - 1):
                weights = weights.unsqueeze(0)
            weights = weights.expand_as(loss)

            loss = loss * weights

        return loss.mean()

    def _calculate_temporal_smoothness_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Calculate temporal smoothness regularization."""
        if predictions.dim() == 4:  # Multi-ship case
            time_dim = 2
        else:  # Single ship case
            time_dim = 1

        if predictions.size(time_dim) < 2:
            return torch.tensor(0.0, device=predictions.device)

        # Calculate temporal differences
        if predictions.dim() == 4:
            diffs = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
        else:
            diffs = predictions[:, 1:, :] - predictions[:, :-1, :]

        smoothness_loss = torch.mean(diffs ** 2)

        return smoothness_loss

    def _calculate_spatial_consistency_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Calculate spatial consistency regularization for multi-ship scenarios."""
        if predictions.dim() != 4:  # Only applies to multi-ship case
            return torch.tensor(0.0, device=predictions.device)

        batch_size, n_ships, forecast_steps, output_dim = predictions.shape

        if n_ships < 2:
            return torch.tensor(0.0, device=predictions.device)

        # Calculate pairwise distances between ship predictions
        spatial_loss = 0.0
        count = 0

        for i in range(n_ships):
            for j in range(i + 1, n_ships):
                # Distance between ship i and j predictions
                ship_i_pred = predictions[:, i, :, :2]  # (batch_size, forecast_steps, 2)
                ship_j_pred = predictions[:, j, :, :2]  # (batch_size, forecast_steps, 2)

                distances = torch.norm(ship_i_pred - ship_j_pred, dim=-1)  # (batch_size, forecast_steps)

                # Encourage smooth distance changes over time
                if forecast_steps > 1:
                    distance_diffs = distances[:, 1:] - distances[:, :-1]
                    spatial_loss += torch.mean(distance_diffs ** 2)
                    count += 1

        return spatial_loss / max(count, 1)


if __name__ == "__main__":
    # Example usage
    config = {
        'model': {
            'architecture': {
                'd_input': 5,
                'd_model': 128,
                'n_heads': 8,
                'n_layers': 4,
                'd_ff': 512,
                'dropout': 0.1,
                'forecast_steps': 30,
                'output_dim': 2,
                'max_distance': 100.0,
                'max_ships': 1
            }
        }
    }

    # Create model
    model = create_stt_model(config)
    print(f"Model info: {model.get_model_info()}")

    # Test forward pass
    batch_size, seq_len = 16, 60
    x = torch.randn(batch_size, seq_len, 5)  # Single ship case
    y = torch.randn(batch_size, 30, 2)

    model.eval()
    with torch.no_grad():
        predictions = model(x)
        print(f"Predictions shape: {predictions.shape}")

    # Test multi-ship case
    x_multi = torch.randn(batch_size, seq_len, 3, 5)  # 3 ships
    y_multi = torch.randn(batch_size, 3, 30, 2)

    config['model']['architecture']['max_ships'] = 3
    model_multi = create_stt_model(config)

    with torch.no_grad():
        predictions_multi = model_multi(x_multi)
        print(f"Multi-ship predictions shape: {predictions_multi.shape}")

    # Test loss
    loss_fn = STTLoss('smooth_l1', 'linear')
    loss = loss_fn(predictions, y)
    print(f"Loss: {loss.item():.4f}")

    loss_multi = loss_fn(predictions_multi, y_multi)
    print(f"Multi-ship loss: {loss_multi.item():.4f}")