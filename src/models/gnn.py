"""
Graph Neural Network (GNN) model for ship trajectory prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class GraphConvLayer(nn.Module):
    """Graph convolution layer."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        """
        Initialize graph convolution layer.

        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            dropout: Dropout probability
        """
        super(GraphConvLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Linear transformations
        self.linear_self = nn.Linear(in_dim, out_dim)
        self.linear_neighbor = nn.Linear(in_dim, out_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Activation
        self.activation = nn.ReLU()

        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, node_features: torch.Tensor,
                edge_indices: torch.Tensor,
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for graph convolution.

        Args:
            node_features: Node features (batch_size, n_nodes, in_dim)
            edge_indices: Edge indices (2, n_edges)
            edge_weights: Edge weights (n_edges,)

        Returns:
            Updated node features (batch_size, n_nodes, out_dim)
        """
        batch_size, n_nodes, _ = node_features.shape

        # Self transformation
        self_features = self.linear_self(node_features)  # (batch_size, n_nodes, out_dim)

        # Neighbor aggregation
        neighbor_features = self._aggregate_neighbors(
            node_features, edge_indices, edge_weights
        )  # (batch_size, n_nodes, in_dim)

        neighbor_features = self.linear_neighbor(neighbor_features)  # (batch_size, n_nodes, out_dim)

        # Combine self and neighbor features
        output = self_features + neighbor_features

        # Apply activation and normalization
        output = self.layer_norm(self.activation(output))

        return self.dropout(output)

    def _aggregate_neighbors(self, node_features: torch.Tensor,
                           edge_indices: torch.Tensor,
                           edge_weights: Optional[torch.Tensor]) -> torch.Tensor:
        """Aggregate neighbor features."""
        batch_size, n_nodes, in_dim = node_features.shape
        device = node_features.device

        # Initialize aggregated features
        aggregated = torch.zeros_like(node_features)

        if edge_indices.size(1) == 0:  # No edges
            return aggregated

        source_idx, target_idx = edge_indices[0], edge_indices[1]

        for b in range(batch_size):
            # Get source features for this batch
            source_features = node_features[b, source_idx, :]  # (n_edges, in_dim)

            # Apply edge weights if provided
            if edge_weights is not None:
                source_features = source_features * edge_weights.unsqueeze(-1)

            # Aggregate by target nodes
            aggregated[b].index_add_(0, target_idx, source_features)

        return aggregated


class TemporalGraphConv(nn.Module):
    """Temporal graph convolution for sequential data."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initialize temporal graph convolution.

        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GCN layers
            dropout: Dropout probability
        """
        super(TemporalGraphConv, self).__init__()

        self.num_layers = num_layers

        # Graph convolution layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layer = GraphConvLayer(in_dim, hidden_dim, dropout)
            else:
                layer = GraphConvLayer(hidden_dim, hidden_dim, dropout)
            self.gcn_layers.append(layer)

        # Temporal aggregation
        self.temporal_weight = nn.Parameter(torch.ones(1))

    def forward(self, node_features: torch.Tensor, edge_indices: torch.Tensor,
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for temporal graph convolution.

        Args:
            node_features: Node features (batch_size, seq_len, n_nodes, in_dim)
            edge_indices: Edge indices (2, n_edges)
            edge_weights: Edge weights (seq_len, n_edges)

        Returns:
            Updated features (batch_size, seq_len, n_nodes, hidden_dim)
        """
        batch_size, seq_len, n_nodes, _ = node_features.shape

        outputs = []

        for t in range(seq_len):
            # Current time step features
            current_features = node_features[:, t, :, :]  # (batch_size, n_nodes, in_dim)

            # Current edge weights
            current_weights = edge_weights[t] if edge_weights is not None else None

            # Apply graph convolutions
            for layer in self.gcn_layers:
                current_features = layer(current_features, edge_indices, current_weights)

            outputs.append(current_features.unsqueeze(1))

        # Stack temporal outputs
        output = torch.cat(outputs, dim=1)  # (batch_size, seq_len, n_nodes, hidden_dim)

        return output


class ShipGraphBuilder(nn.Module):
    """Build graph structure for ship trajectories."""

    def __init__(self, max_distance: float = 5.0, k_neighbors: int = 5):
        """
        Initialize graph builder.

        Args:
            max_distance: Maximum distance for edge creation (nautical miles)
            k_neighbors: Maximum number of neighbors per ship
        """
        super(ShipGraphBuilder, self).__init__()

        self.max_distance = max_distance
        self.k_neighbors = k_neighbors

    def build_graph(self, positions: torch.Tensor,
                   velocities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph structure from ship positions and velocities.

        Args:
            positions: Ship positions (batch_size, n_ships, 2) in lat/lon
            velocities: Ship velocities (batch_size, n_ships, 2) in lat/lon per minute

        Returns:
            edge_indices: Edge indices (2, n_edges)
            edge_weights: Edge weights (n_edges,)
        """
        batch_size, n_ships, _ = positions.shape
        device = positions.device

        edge_indices_list = []
        edge_weights_list = []

        # Convert to approximate distances (simplified)
        lat_to_nm = 60.0  # 1 degree latitude ≈ 60 nautical miles

        for b in range(batch_size):
            pos = positions[b]  # (n_ships, 2)
            vel = velocities[b]  # (n_ships, 2)

            # Calculate pairwise distances
            pos_expanded = pos.unsqueeze(1)  # (n_ships, 1, 2)
            pos_tiled = pos.unsqueeze(0)     # (1, n_ships, 2)

            dist_matrix = torch.norm(
                (pos_expanded - pos_tiled) * lat_to_nm, dim=2
            )  # (n_ships, n_ships)

            # Calculate relative velocities
            vel_expanded = vel.unsqueeze(1)  # (n_ships, 1, 2)
            vel_tiled = vel.unsqueeze(0)     # (1, n_ships, 2)

            rel_vel = vel_expanded - vel_tiled  # (n_ships, n_ships, 2)
            rel_speed = torch.norm(rel_vel, dim=2)  # (n_ships, n_ships)

            # Create edges based on distance and relative motion
            for i in range(n_ships):
                # Get distances to all other ships
                distances = dist_matrix[i]

                # Filter by maximum distance
                valid_mask = (distances < self.max_distance) & (distances > 0)
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()

                if len(valid_indices) > 0:
                    # Select k nearest neighbors
                    k = min(self.k_neighbors, len(valid_indices))
                    _, top_k_idx = torch.topk(-distances[valid_indices], k)
                    selected_neighbors = valid_indices[top_k_idx]

                    # Create edges
                    source_nodes = torch.full((k,), i, device=device)
                    target_nodes = selected_neighbors

                    edge_indices_list.extend([
                        torch.stack([source_nodes, target_nodes], dim=0)
                    ])

                    # Calculate edge weights based on distance and relative motion
                    edge_distances = distances[selected_neighbors]
                    edge_rel_speeds = rel_speed[i, selected_neighbors]

                    # Weight = 1 / (distance + epsilon) * relative_speed_factor
                    distance_weight = 1.0 / (edge_distances + 0.1)
                    motion_weight = torch.exp(-edge_rel_speeds)  # Closer relative speeds get higher weights

                    weights = distance_weight * motion_weight
                    edge_weights_list.append(weights)

        if edge_indices_list:
            edge_indices = torch.cat(edge_indices_list, dim=1)
            edge_weights = torch.cat(edge_weights_list, dim=0)
        else:
            # No edges
            edge_indices = torch.empty((2, 0), device=device, dtype=torch.long)
            edge_weights = torch.empty((0,), device=device)

        return edge_indices, edge_weights


class TrajGNN(nn.Module):
    """Graph Neural Network for ship trajectory prediction."""

    def __init__(self, config: Dict):
        """
        Initialize GNN model.

        Args:
            config: Model configuration dictionary
        """
        super(TrajGNN, self).__init__()

        self.config = config
        arch_config = config['model']['architecture']

        # Architecture parameters
        self.d_input = arch_config['d_input']
        self.hidden_dim = arch_config['hidden_dim']
        self.gnn_layers = arch_config['gnn_layers']
        self.dropout = arch_config['dropout']

        # Graph parameters
        self.max_distance = arch_config.get('max_distance', 5.0)
        self.k_neighbors = arch_config.get('k_neighbors', 5)

        # Output parameters
        self.forecast_steps = arch_config['forecast_steps']
        self.output_dim = arch_config['output_dim']

        # Temporal encoding
        self.use_temporal_encoding = arch_config.get('use_temporal_encoding', True)
        self.temporal_dim = arch_config.get('temporal_dim', 32)

        # Build model
        self._build_model()

    def _build_model(self):
        """Build GNN architecture."""
        # Input projection
        input_dim = self.d_input
        if self.use_temporal_encoding:
            input_dim += self.temporal_dim
            # Temporal encoding
            self.temporal_embedding = nn.Embedding(512, self.temporal_dim)

        self.input_projection = nn.Linear(input_dim, self.hidden_dim)

        # Graph builder
        self.graph_builder = ShipGraphBuilder(self.max_distance, self.k_neighbors)

        # Temporal graph convolution
        self.temporal_gcn = TemporalGraphConv(
            self.hidden_dim, self.hidden_dim, self.gnn_layers, self.dropout
        )

        # Temporal aggregation (LSTM for sequence modeling)
        self.temporal_lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            dropout=self.dropout,
            batch_first=True
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.forecast_steps * self.output_dim)
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
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x: torch.Tensor, ship_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, n_ships, d_input)
            ship_ids: Ship identifiers for multi-ship scenarios

        Returns:
            Predicted trajectories (batch_size, n_ships, forecast_steps, output_dim)
        """
        # Handle single ship case by adding ship dimension
        if x.dim() == 3:
            x = x.unsqueeze(2)  # (batch_size, seq_len, 1, d_input)

        batch_size, seq_len, n_ships, _ = x.shape
        device = x.device

        # Add temporal encoding
        if self.use_temporal_encoding:
            time_indices = torch.arange(seq_len, device=device)
            time_emb = self.temporal_embedding(time_indices)  # (seq_len, temporal_dim)
            time_emb = time_emb.unsqueeze(0).unsqueeze(2).expand(
                batch_size, seq_len, n_ships, -1
            )  # (batch_size, seq_len, n_ships, temporal_dim)

            x = torch.cat([x, time_emb], dim=-1)  # (batch_size, seq_len, n_ships, d_input + temporal_dim)

        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, n_ships, hidden_dim)

        # Build graph structure for each time step
        edge_indices_list = []
        edge_weights_list = []

        for t in range(seq_len):
            positions = x[:, t, :, :2]  # Assume first 2 dims are lat/lon
            velocities = x[:, t, :, 2:4] if x.size(-1) > 2 else torch.zeros_like(positions)

            # Average over batch for graph structure (simplification)
            avg_pos = positions.mean(dim=0)  # (n_ships, 2)
            avg_vel = velocities.mean(dim=0)  # (n_ships, 2)

            edge_indices, edge_weights = self.graph_builder.build_graph(
                avg_pos.unsqueeze(0), avg_vel.unsqueeze(0)
            )

            edge_indices_list.append(edge_indices)
            edge_weights_list.append(edge_weights)

        # Stack edge weights
        max_edges = max(len(w) for w in edge_weights_list) if edge_weights_list else 0

        if max_edges > 0:
            # Pad edge weights to same length
            padded_weights = []
            for weights in edge_weights_list:
                if len(weights) < max_edges:
                    padding = torch.zeros(max_edges - len(weights), device=device)
                    weights = torch.cat([weights, padding])
                padded_weights.append(weights)

            edge_weights_tensor = torch.stack(padded_weights)  # (seq_len, max_edges)

            # Use first time step's edge structure for all time steps (simplification)
            edge_indices = edge_indices_list[0] if edge_indices_list else torch.empty((2, 0), device=device, dtype=torch.long)
        else:
            edge_indices = torch.empty((2, 0), device=device, dtype=torch.long)
            edge_weights_tensor = None

        # Apply temporal graph convolution
        if edge_indices.size(1) > 0:
            graph_features = self.temporal_gcn(x, edge_indices, edge_weights_tensor)
        else:
            # No graph structure, just pass through
            graph_features = x

        # Global pooling over ships (for single ship prediction)
        if n_ships == 1:
            graph_features = graph_features.squeeze(2)  # (batch_size, seq_len, hidden_dim)
        else:
            # Mean pooling over ships
            graph_features = graph_features.mean(dim=2)  # (batch_size, seq_len, hidden_dim)

        # Temporal modeling with LSTM
        temporal_output, _ = self.temporal_lstm(graph_features)  # (batch_size, seq_len, hidden_dim)

        # Use last time step for prediction
        final_representation = temporal_output[:, -1, :]  # (batch_size, hidden_dim)

        # Generate predictions
        predictions = self.output_head(final_representation)  # (batch_size, forecast_steps * output_dim)
        predictions = predictions.view(batch_size, self.forecast_steps, self.output_dim)

        return predictions

    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'TrajGNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': {
                'input_dim': self.d_input,
                'hidden_dim': self.hidden_dim,
                'gnn_layers': self.gnn_layers,
                'max_distance': self.max_distance,
                'k_neighbors': self.k_neighbors,
                'forecast_steps': self.forecast_steps,
                'output_dim': self.output_dim,
                'temporal_encoding': self.use_temporal_encoding
            }
        }


def create_gnn_model(config: Dict, model_type: str = 'basic') -> nn.Module:
    """
    Factory function to create GNN models.

    Args:
        config: Model configuration
        model_type: Type of GNN model

    Returns:
        GNN model instance
    """
    if model_type == 'basic':
        return TrajGNN(config)
    else:
        raise ValueError(f"Unknown GNN model type: {model_type}")


class GNNLoss(nn.Module):
    """Custom loss function for GNN trajectory prediction."""

    def __init__(self, loss_type: str = 'smooth_l1', step_weights: str = 'linear',
                 smoothness_weight: float = 0.01):
        """
        Initialize GNN loss.

        Args:
            loss_type: Base loss type
            step_weights: Step weighting scheme
            smoothness_weight: Weight for smoothness regularization
        """
        super(GNNLoss, self).__init__()

        self.loss_type = loss_type
        self.step_weights = step_weights
        self.smoothness_weight = smoothness_weight

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
        """Calculate total loss."""
        # Base prediction loss
        pred_loss = self._calculate_prediction_loss(predictions, targets)

        # Smoothness regularization
        smoothness_loss = self._calculate_smoothness_loss(predictions)

        # Total loss
        total_loss = pred_loss + self.smoothness_weight * smoothness_loss

        return total_loss

    def _calculate_prediction_loss(self, predictions: torch.Tensor,
                                 targets: torch.Tensor) -> torch.Tensor:
        """Calculate weighted prediction loss."""
        loss = self.base_loss(predictions, targets)
        loss = loss.mean(dim=-1)  # Average over output dimensions

        # Apply step weights
        if self.step_weights == 'linear':
            steps = torch.arange(1, loss.size(1) + 1, device=loss.device, dtype=torch.float)
            weights = steps / steps.sum()
            weights = weights.unsqueeze(0).expand_as(loss)
            loss = loss * weights

        return loss.mean()

    def _calculate_smoothness_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Calculate smoothness regularization."""
        if predictions.size(1) < 2:
            return torch.tensor(0.0, device=predictions.device)

        diffs = predictions[:, 1:, :] - predictions[:, :-1, :]
        smoothness_loss = torch.mean(diffs ** 2)

        return smoothness_loss


if __name__ == "__main__":
    # Example usage
    config = {
        'model': {
            'architecture': {
                'd_input': 5,
                'hidden_dim': 128,
                'gnn_layers': 3,
                'dropout': 0.1,
                'max_distance': 5.0,
                'k_neighbors': 5,
                'forecast_steps': 30,
                'output_dim': 2,
                'use_temporal_encoding': True,
                'temporal_dim': 32
            }
        }
    }

    # Create model
    model = create_gnn_model(config)
    print(f"Model info: {model.get_model_info()}")

    # Test forward pass
    batch_size, seq_len = 16, 60
    x = torch.randn(batch_size, seq_len, 5)  # Single ship case
    y = torch.randn(batch_size, 30, 2)

    model.eval()
    with torch.no_grad():
        predictions = model(x)
        print(f"Predictions shape: {predictions.shape}")

    # Test loss
    loss_fn = GNNLoss('smooth_l1', 'linear')
    loss = loss_fn(predictions, y)
    print(f"Loss: {loss.item():.4f}")