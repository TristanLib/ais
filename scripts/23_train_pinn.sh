#!/bin/bash
# Script to train Physics-Informed Neural Network (PINN) model for ship trajectory prediction

set -e  # Exit on any error

echo "=== Training PINN Model ==="
echo "Starting PINN model training..."

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

# Train PINN model
echo "Training PINN model..."
python3 << 'EOF'
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
from models.pinn import create_pinn_model, PINNLoss, PhysicsWeightScheduler
from eval.metrics import TrajectoryMetrics

# Load configuration
with open('$CONFIG_DIR/model_pinn.yaml', 'r') as f:
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

model = create_pinn_model(config, 'basic')
model = model.to(device)

print(f"Model info: {model.get_model_info()}")

# Physics weights and scheduler
physics_weights = config['model'].get('physics_weights', {
    'kinematic': 1.0,
    'acceleration': 0.5,
    'turning': 0.3,
    'speed': 0.2,
    'continuity': 0.4
})

# Loss function
loss_fn = PINNLoss(
    config['model']['training']['loss_type'],
    physics_weights
)

# Physics weight scheduler
use_scheduling = config['model'].get('physics_scheduling', {}).get('enable', False)
if use_scheduling:
    physics_scheduler = PhysicsWeightScheduler(
        initial_weights=physics_weights,
        schedule_type=config['model']['physics_scheduling'].get('schedule_type', 'linear'),
        warmup_epochs=config['model']['physics_scheduling'].get('warmup_epochs', 20),
        max_epochs=config['model']['training']['max_epochs']
    )
    print("Using physics weight scheduling")

# Optimizer
optimizer_name = config['model']['training']['optimizer']
if optimizer_name == 'adamw':
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['model']['training']['learning_rate'],
        weight_decay=config['model']['training']['weight_decay']
    )
elif optimizer_name == 'adam':
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['model']['training']['learning_rate'],
        weight_decay=config['model']['training']['weight_decay']
    )

# Learning rate scheduler
scheduler_type = config['model']['training']['scheduler']
if scheduler_type == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['model']['training'].get('step_size', 30),
        gamma=config['model']['training'].get('gamma', 0.5)
    )
elif scheduler_type == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['model']['training']['max_epochs']
    )
else:
    scheduler = None

# Curriculum learning
use_curriculum = config.get('training_strategy', {}).get('use_curriculum', False)
if use_curriculum:
    curriculum_stages = config['training_strategy']['curriculum_stages']
    print(f"Using curriculum learning with {len(curriculum_stages)} stages")

# Training loop
print("Starting PINN training...")
max_epochs = config['model']['training']['max_epochs']
best_val_loss = float('inf')
patience = config['model']['training']['early_stopping_patience']
patience_counter = 0

training_start = time.time()
train_losses = []
val_losses = []
physics_losses = []
data_losses = []
learning_rates = []

# Current curriculum stage
current_stage = 0
stage_epochs = 0

for epoch in range(max_epochs):
    # Curriculum learning
    if use_curriculum and current_stage < len(curriculum_stages):
        stage_info = curriculum_stages[current_stage]
        if stage_epochs >= stage_info['epochs']:
            current_stage += 1
            stage_epochs = 0
            if current_stage < len(curriculum_stages):
                print(f"Moving to curriculum stage {current_stage + 1}")

        if current_stage < len(curriculum_stages):
            # Update physics weights for current stage
            current_weights = {}
            for key in physics_weights:
                current_weights[key] = physics_weights[key] * stage_info['physics_weight']
            loss_fn.update_physics_weights(current_weights)

        stage_epochs += 1

    # Physics weight scheduling
    if use_scheduling and not use_curriculum:
        updated_weights = physics_scheduler.step(epoch)
        loss_fn.update_physics_weights(updated_weights)

    # Training
    model.train()
    train_loss = 0.0
    train_physics_loss = 0.0
    train_data_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass with physics terms
        predictions, physics_terms = model(data, return_physics_terms=True)

        # Calculate loss
        total_loss, loss_components = loss_fn(predictions, target, physics_terms)

        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        train_physics_loss += loss_components.get('physics_loss', torch.tensor(0.0)).item()
        train_data_loss += loss_components.get('data_loss', torch.tensor(0.0)).item()

    train_loss /= len(train_loader)
    train_physics_loss /= len(train_loader)
    train_data_loss /= len(train_loader)

    train_losses.append(train_loss)
    physics_losses.append(train_physics_loss)
    data_losses.append(train_data_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    val_physics_loss = 0.0
    val_data_loss = 0.0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            predictions, physics_terms = model(data, return_physics_terms=True)
            total_loss, loss_components = loss_fn(predictions, target, physics_terms)

            val_loss += total_loss.item()
            val_physics_loss += loss_components.get('physics_loss', torch.tensor(0.0)).item()
            val_data_loss += loss_components.get('data_loss', torch.tensor(0.0)).item()

    val_loss /= len(val_loader)
    val_physics_loss /= len(val_loader)
    val_data_loss /= len(val_loader)

    val_losses.append(val_loss)

    # Learning rate scheduling
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)

    if scheduler is not None:
        scheduler.step()

    print(f"Epoch {epoch+1}/{max_epochs}: Total Loss: {train_loss:.4f}, Data Loss: {train_data_loss:.4f}, Physics Loss: {train_physics_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")

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
        }, '$OUTPUT_DIR/models/pinn_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

training_time = time.time() - training_start
print(f"PINN training completed in {training_time:.2f} seconds")

# Load best model for evaluation
checkpoint = torch.load('$OUTPUT_DIR/models/pinn_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate model
print("Evaluating PINN model...")
all_predictions = []
all_targets = []

inference_start = time.time()
with torch.no_grad():
    for data, target in val_loader:
        data = data.to(device)

        # Use standard forward pass for inference
        output = model(data, return_physics_terms=False)

        all_predictions.append(output.cpu().numpy())
        all_targets.append(target.numpy())

inference_time = time.time() - inference_start

pinn_predictions = np.concatenate(all_predictions, axis=0)
pinn_targets = np.concatenate(all_targets, axis=0)

print(f"PINN inference time: {inference_time:.2f} seconds")
print(f"PINN predictions shape: {pinn_predictions.shape}")

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
pinn_results = metrics_calc.evaluate_model(pinn_predictions, pinn_targets)

print("PINN Model Results:")
for metric, value in pinn_results['overall_metrics'].items():
    print(f"  {metric.upper()}: {value:.4f}")

# Save results
results_path = '$OUTPUT_DIR/models/pinn_results.json'
with open(results_path, 'w') as f:
    json.dump({
        'model_type': 'pinn',
        'training_time': training_time,
        'inference_time': inference_time,
        'metrics': pinn_results,
        'config': config,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'physics_losses': physics_losses,
            'data_losses': data_losses,
            'learning_rates': learning_rates,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
    }, f, indent=2, default=str)

print(f"Saved PINN results to: {results_path}")

# Save training history plot
import matplotlib.pyplot as plt

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Total loss plot
ax1.plot(train_losses, label='Training Loss')
ax1.plot(val_losses, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('PINN Total Loss History')
ax1.legend()
ax1.grid(True)

# Component losses
ax2.plot(data_losses, label='Data Loss')
ax2.plot(physics_losses, label='Physics Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('PINN Loss Components')
ax2.legend()
ax2.grid(True)

# Learning rate plot
ax3.plot(learning_rates, label='Learning Rate')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Learning Rate')
ax3.set_title('Learning Rate Schedule')
ax3.legend()
ax3.grid(True)
ax3.set_yscale('log')

# Physics vs Data loss ratio
if len(physics_losses) > 0 and all(p > 0 for p in physics_losses):
    ratio = [d/p if p > 0 else 0 for d, p in zip(data_losses, physics_losses)]
    ax4.plot(ratio, label='Data/Physics Loss Ratio')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Ratio')
    ax4.set_title('Data to Physics Loss Ratio')
    ax4.legend()
    ax4.grid(True)

plt.tight_layout()
plt.savefig('$OUTPUT_DIR/figures/pinn_training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved training history plot")

# Export model for deployment (without physics terms for faster inference)
torch.jit.save(torch.jit.script(model.cpu()), '$OUTPUT_DIR/models/pinn_scripted.pt')
print("Saved TorchScript model for deployment")

EOF

echo "=== PINN Training Complete ==="
echo "Models saved to: $OUTPUT_DIR/models/"
echo "Results saved to: $OUTPUT_DIR/models/"
echo "Figures saved to: $OUTPUT_DIR/figures/"
echo ""
echo "Next steps:"
echo "1. Evaluate all models: bash scripts/30_eval_all.sh"
echo "2. Run collision avoidance simulation: bash scripts/40_sim_avoidance.sh"