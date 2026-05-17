"""
Machine learning models for ship trajectory prediction.
"""

from .baselines import (
    BaseTrajectoryPredictor,
    ConstantVelocityPredictor,
    ConstantAccelerationPredictor,
    LinearExtrapolationPredictor,
    MovingAveragePredictor,
    create_baseline_predictor,
    BaselineEnsemble
)

from .lstm import (
    TrajLSTM,
    LSTMWithAttention,
    create_lstm_model,
    LSTMLoss
)

from .transformer import (
    TrajTransformer,
    TransformerDecoderModel,
    create_transformer_model,
    TransformerLoss,
    PositionalEncoding,
    LearnablePositionalEncoding
)

from .gnn import (
    TrajGNN,
    create_gnn_model,
    GNNLoss,
    GraphConvLayer,
    TemporalGraphConv,
    ShipGraphBuilder
)

from .stt import (
    TrajSTT,
    create_stt_model,
    STTLoss,
    SpatialPositionalEncoding,
    TemporalAttention,
    SpatialAttention,
    SpatioTemporalBlock
)

from .pinn import (
    TrajPINN,
    create_pinn_model,
    PINNLoss,
    PhysicsModule,
    PhysicsWeightScheduler
)

__all__ = [
    # Baselines
    'BaseTrajectoryPredictor',
    'ConstantVelocityPredictor',
    'ConstantAccelerationPredictor',
    'LinearExtrapolationPredictor',
    'MovingAveragePredictor',
    'create_baseline_predictor',
    'BaselineEnsemble',

    # LSTM
    'TrajLSTM',
    'LSTMWithAttention',
    'create_lstm_model',
    'LSTMLoss',

    # Transformer
    'TrajTransformer',
    'TransformerDecoderModel',
    'create_transformer_model',
    'TransformerLoss',
    'PositionalEncoding',
    'LearnablePositionalEncoding',

    # GNN
    'TrajGNN',
    'create_gnn_model',
    'GNNLoss',
    'GraphConvLayer',
    'TemporalGraphConv',
    'ShipGraphBuilder',

    # STT
    'TrajSTT',
    'create_stt_model',
    'STTLoss',
    'SpatialPositionalEncoding',
    'TemporalAttention',
    'SpatialAttention',
    'SpatioTemporalBlock',

    # PINN
    'TrajPINN',
    'create_pinn_model',
    'PINNLoss',
    'PhysicsModule',
    'PhysicsWeightScheduler'
]
