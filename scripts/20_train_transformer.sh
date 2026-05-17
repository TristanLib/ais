#!/bin/bash
# Script to train Transformer model for ship trajectory prediction

set -e  # Exit on any error

echo "=== Training Transformer Model ==="
echo "Starting Transformer model training..."

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
CONFIG_DIR="$PROJECT_ROOT/configs"
OUTPUT_DIR="$PROJECT_ROOT/outputs"

echo "Project root: $PROJECT_ROOT"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Activate conda environment
if command -v conda &> /dev/null; then
    if conda env list | grep -q "ship-prediction"; then
        echo "Activating conda environment: ship-prediction"
        eval "$(conda shell.bash hook)"
        conda activate ship-prediction
    fi
fi

# Check if processed data exists
if [ ! -f "$DATA_DIR/processed/train_samples.pkl" ]; then
    echo "Error: Processed data not found. Please run 00_make_dataset.sh first."
    exit 1
fi

# Train Transformer model
echo "Training Transformer model..."
python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')

import yaml
import numpy as np
import pickle
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from dataio.slicing import create_numpy_arrays
from models.transformer import create_transformer_model, TransformerLoss
from eval.metrics import TrajectoryMetrics

# Load configuration
with open('$CONFIG_DIR/model_transformer.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load data
print("Loading data...")
with open('$DATA_DIR/processed/train_samples.pkl', 'rb') as f:
    train_samples = pickle.load(f)

with open('$DATA_DIR/processed/val_samples.pkl', 'rb') as f:
    val_samples = pickle.load(f)

# Convert to numpy arrays
X_train, y_train = create_numpy_arrays(train_samples)
X_val, y_val = create_numpy_arrays(val_samples)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

batch_size = config['model']['training']['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Created data loaders with batch size {batch_size}")

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = create_transformer_model(config, 'encoder')
model = model.to(device)

print(f"Model info: {model.get_model_info()}")

# Loss and optimizer
loss_fn = TransformerLoss(
    config['model']['training']['loss_type'],
    config['model']['training']['step_weights'],
    config['model']['training'].get('coord_smoothness_weight', 0.01),
    config['model']['training'].get('direction_consistency_weight', 0.005)
)

optimizer_name = config['model']['training']['optimizer']
if optimizer_name == 'adamw':
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['model']['training']['learning_rate'],
        weight_decay=config['model']['training']['weight_decay']
    )
else:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['model']['training']['learning_rate'],
        weight_decay=config['model']['training']['weight_decay']
    )

# Learning rate scheduler
scheduler_type = config['model']['training']['scheduler']
if scheduler_type == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['model']['training']['max_epochs']
    )
elif scheduler_type == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )
else:
    scheduler = None

# Warmup epochs
warmup_epochs = config['model']['training']['warmup_epochs']

# Mixed precision training
use_amp = config['model'].get('use_amp', False)
if use_amp:
    scaler = torch.cuda.amp.GradScaler()
    print("Using mixed precision training")

# Training loop
print("Starting Transformer training...")
max_epochs = config['model']['training']['max_epochs']
best_val_loss = float('inf')
patience = config['model']['training']['early_stopping_patience']
patience_counter = 0

training_start = time.time()
train_losses = []
val_losses = []
learning_rates = []

for epoch in range(max_epochs):
    # Warmup learning rate
    if epoch < warmup_epochs:
        lr = config['model']['training']['learning_rate'] * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Training
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = loss_fn(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = loss_fn(output, target)
            else:
                output = model(data)
                loss = loss_fn(output, target)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Learning rate scheduling
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)

    if scheduler is not None and epoch >= warmup_epochs:
        scheduler.step()

    print(f"Epoch {epoch+1}/{max_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
            'val_loss': val_loss
        }, '$OUTPUT_DIR/models/transformer_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

training_time = time.time() - training_start
print(f"Transformer training completed in {training_time:.2f} seconds")

# Load best model for evaluation
checkpoint = torch.load('$OUTPUT_DIR/models/transformer_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate model
print("Evaluating Transformer model...")
all_predictions = []
all_targets = []

inference_start = time.time()
with torch.no_grad():
    for data, target in val_loader:
        data = data.to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(data)
        else:
            output = model(data)

        all_predictions.append(output.cpu().numpy())
        all_targets.append(target.numpy())

inference_time = time.time() - inference_start

transformer_predictions = np.concatenate(all_predictions, axis=0)
transformer_targets = np.concatenate(all_targets, axis=0)

print(f"Transformer inference time: {inference_time:.2f} seconds")
print(f"Transformer predictions shape: {transformer_predictions.shape}")

# Calculate metrics
eval_config = {
    'evaluation': {
        'prediction': {
            'metrics': ['ade', 'fde', 'rmse', 'mae'],
            'horizons': [5, 10, 20, 30],
            'confidence_intervals': [0.95]
        }
    }
}

metrics_calc = TrajectoryMetrics(eval_config)
transformer_results = metrics_calc.evaluate_model(transformer_predictions, transformer_targets)

print("Transformer Model Results:")
for metric, value in transformer_results['overall_metrics'].items():
    print(f"  {metric.upper()}: {value:.4f}")

# Save results
results_path = '$OUTPUT_DIR/models/transformer_results.json'
with open(results_path, 'w') as f:
    json.dump({
        'model_type': 'transformer',
        'training_time': training_time,
        'inference_time': inference_time,
        'metrics': transformer_results,
        'config': config,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
    }, f, indent=2, default=str)

print(f"Saved Transformer results to: {results_path}")

# Save training history plot
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Loss plot
ax1.plot(train_losses, label='Training Loss')
ax1.plot(val_losses, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Transformer Training History')
ax1.legend()
ax1.grid(True)

# Learning rate plot
ax2.plot(learning_rates, label='Learning Rate')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Schedule')
ax2.legend()
ax2.grid(True)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('$OUTPUT_DIR/figures/transformer_training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved training history plot")

# Export model for deployment
torch.jit.save(torch.jit.script(model.cpu()), '$OUTPUT_DIR/models/transformer_scripted.pt')
print("Saved TorchScript model for deployment")

# Try to export ONNX (if supported)
try:
    import torch.onnx
    dummy_input = torch.randn(1, 60, 5)  # batch_size=1, seq_len=60, features=5
    torch.onnx.export(
        model,
        dummy_input,
        '$OUTPUT_DIR/models/transformer_model.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("Saved ONNX model for deployment")
except Exception as e:
    print(f"ONNX export failed: {e}")
EOF

echo "=== Transformer Training Complete ==="
echo "Models saved to: $OUTPUT_DIR/models/"
echo "Results saved to: $OUTPUT_DIR/models/"
echo "Figures saved to: $OUTPUT_DIR/figures/"
echo ""
echo "Next steps:"
echo "1. Evaluate all models: bash scripts/30_eval_all.sh"
echo "2. Run collision avoidance simulation: bash scripts/40_sim_avoidance.sh"
