#!/bin/bash
# Script to train baseline models (CV, LSTM) for ship trajectory prediction

set -e  # Exit on any error

echo "=== Training Baseline Models ==="
echo "Starting baseline model training..."

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
CONFIG_DIR="$PROJECT_ROOT/configs"
OUTPUT_DIR="$PROJECT_ROOT/outputs"

echo "Project root: $PROJECT_ROOT"

# Create output directories
mkdir -p "$OUTPUT_DIR"/{logs,models,figures}

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

# Step 1: Train constant velocity baseline
echo "Step 1: Training Constant Velocity baseline..."
python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')

import yaml
import numpy as np
import pickle
import json
import time
from pathlib import Path

from dataio.slicing import create_numpy_arrays
from models.baselines import create_baseline_predictor
from eval.metrics import TrajectoryMetrics

# Load configuration
with open('$CONFIG_DIR/data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

# Load data
print("Loading training data...")
with open('$DATA_DIR/processed/train_samples.pkl', 'rb') as f:
    train_samples = pickle.load(f)

with open('$DATA_DIR/processed/val_samples.pkl', 'rb') as f:
    val_samples = pickle.load(f)

# Convert to numpy arrays
X_train, y_train = create_numpy_arrays(train_samples)
X_val, y_val = create_numpy_arrays(val_samples)

print(f"Training data: X shape {X_train.shape}, y shape {y_train.shape}")
print(f"Validation data: X shape {X_val.shape}, y shape {y_val.shape}")

# Model configuration
model_config = {
    'dt_minutes': data_config['data']['processing']['dt_minutes'],
    'forecast_steps': data_config['data']['processing']['forecast_minutes'],
}

# Train Constant Velocity model
print("Training Constant Velocity model...")
cv_model = create_baseline_predictor('constant_velocity', model_config)

start_time = time.time()
cv_model.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"CV model training time: {training_time:.2f} seconds")

# Evaluate on validation set
print("Evaluating CV model...")
start_time = time.time()
cv_predictions = cv_model.predict(X_val)
inference_time = time.time() - start_time

print(f"CV model inference time: {inference_time:.2f} seconds")
print(f"CV predictions shape: {cv_predictions.shape}")

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
cv_results = metrics_calc.evaluate_model(cv_predictions, y_val)

print("CV Model Results:")
for metric, value in cv_results['overall_metrics'].items():
    print(f"  {metric.upper()}: {value:.4f}")

# Save model and results
model_path = '$OUTPUT_DIR/models/cv_baseline.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(cv_model, f)

results_path = '$OUTPUT_DIR/models/cv_baseline_results.json'
with open(results_path, 'w') as f:
    json.dump({
        'model_type': 'constant_velocity',
        'training_time': training_time,
        'inference_time': inference_time,
        'metrics': cv_results,
        'config': model_config
    }, f, indent=2, default=str)

print(f"Saved CV model to: {model_path}")
print(f"Saved CV results to: {results_path}")
EOF

# Step 2: Train LSTM baseline
echo "Step 2: Training LSTM baseline..."
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
from models.lstm import create_lstm_model, LSTMLoss
from eval.metrics import TrajectoryMetrics

# Load configuration
with open('$CONFIG_DIR/model_lstm.yaml', 'r') as f:
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

model = create_lstm_model(config, 'basic')
model = model.to(device)

print(f"Model info: {model.get_model_info()}")

# Loss and optimizer
loss_fn = LSTMLoss(
    config['model']['training']['loss_type'],
    config['model']['training']['step_weights']
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config['model']['training']['learning_rate'],
    weight_decay=config['model']['training']['weight_decay']
)

# Training loop
print("Starting LSTM training...")
max_epochs = config['model']['training']['max_epochs']
best_val_loss = float('inf')
patience = config['model']['training']['early_stopping_patience']
patience_counter = 0

training_start = time.time()
train_losses = []
val_losses = []

for epoch in range(max_epochs):
    # Training
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
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
            output = model(data)
            val_loss += loss_fn(output, target).item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{max_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), '$OUTPUT_DIR/models/lstm_baseline_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

training_time = time.time() - training_start
print(f"LSTM training completed in {training_time:.2f} seconds")

# Load best model for evaluation
model.load_state_dict(torch.load('$OUTPUT_DIR/models/lstm_baseline_best.pth'))
model.eval()

# Evaluate model
print("Evaluating LSTM model...")
all_predictions = []
all_targets = []

inference_start = time.time()
with torch.no_grad():
    for data, target in val_loader:
        data = data.to(device)
        output = model(data)
        all_predictions.append(output.cpu().numpy())
        all_targets.append(target.numpy())

inference_time = time.time() - inference_start

lstm_predictions = np.concatenate(all_predictions, axis=0)
lstm_targets = np.concatenate(all_targets, axis=0)

print(f"LSTM inference time: {inference_time:.2f} seconds")
print(f"LSTM predictions shape: {lstm_predictions.shape}")

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
lstm_results = metrics_calc.evaluate_model(lstm_predictions, lstm_targets)

print("LSTM Model Results:")
for metric, value in lstm_results['overall_metrics'].items():
    print(f"  {metric.upper()}: {value:.4f}")

# Save results
results_path = '$OUTPUT_DIR/models/lstm_baseline_results.json'
with open(results_path, 'w') as f:
    json.dump({
        'model_type': 'lstm',
        'training_time': training_time,
        'inference_time': inference_time,
        'metrics': lstm_results,
        'config': config,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
    }, f, indent=2, default=str)

print(f"Saved LSTM results to: {results_path}")

# Save training history plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM Training History')
plt.legend()
plt.grid(True)
plt.savefig('$OUTPUT_DIR/figures/lstm_training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved training history plot")
EOF

echo "=== Baseline Training Complete ==="
echo "Models saved to: $OUTPUT_DIR/models/"
echo "Results saved to: $OUTPUT_DIR/models/"
echo "Figures saved to: $OUTPUT_DIR/figures/"
echo ""
echo "Next steps:"
echo "1. Train Transformer: bash scripts/20_train_transformer.sh"
echo "2. Evaluate all models: bash scripts/30_eval_all.sh"
