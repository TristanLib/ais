#!/bin/bash
# Script to evaluate all models and generate comparison report

set -e  # Exit on any error

echo "=== Model Evaluation and Comparison ==="
echo "Starting comprehensive model evaluation..."

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

# Check if models exist
if [ ! -f "$OUTPUT_DIR/models/cv_baseline.pkl" ] || [ ! -f "$OUTPUT_DIR/models/lstm_baseline_best.pth" ] || [ ! -f "$OUTPUT_DIR/models/transformer_best.pth" ]; then
    echo "Error: Not all baseline models found. Please train models first:"
    echo "  - Run: bash scripts/10_train_baselines.sh"
    echo "  - Run: bash scripts/20_train_transformer.sh"
    exit 1
fi

# Check for new models (optional - they might not be trained yet)
NEW_MODELS_AVAILABLE=false
if [ -f "$OUTPUT_DIR/models/gnn_best.pth" ] && [ -f "$OUTPUT_DIR/models/stt_best.pth" ] && [ -f "$OUTPUT_DIR/models/pinn_best.pth" ]; then
    NEW_MODELS_AVAILABLE=true
    echo "New models (GNN, STT, PINN) found - including in evaluation"
else
    echo "New models not found - evaluating baseline models only"
    echo "To include new models, run:"
    echo "  - bash scripts/21_train_gnn.sh"
    echo "  - bash scripts/22_train_stt.sh"
    echo "  - bash scripts/23_train_pinn.sh"
fi

# Comprehensive evaluation
echo "Running comprehensive evaluation on test set..."
python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')

import yaml
import numpy as np
import pickle
import json
import time
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from dataio.slicing import create_numpy_arrays
from models.baselines import create_baseline_predictor
from models.lstm import create_lstm_model
from models.transformer import create_transformer_model
from eval.metrics import TrajectoryMetrics, StatisticalTesting, create_evaluation_report

# Import new models if available
try:
    from models.gnn import create_gnn_model
    from models.stt import create_stt_model
    from models.pinn import create_pinn_model
    NEW_MODELS_IMPORTED = True
except ImportError as e:
    print(f"Warning: Could not import new models: {e}")
    NEW_MODELS_IMPORTED = False

# Load test data
print("Loading test data...")
with open('$DATA_DIR/processed/test_samples.pkl', 'rb') as f:
    test_samples = pickle.load(f)

X_test, y_test = create_numpy_arrays(test_samples)
print(f"Test data: X shape {X_test.shape}, y shape {y_test.shape}")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Evaluation configuration
eval_config = {
    'evaluation': {
        'prediction': {
            'metrics': ['ade', 'fde', 'rmse', 'mae'],
            'horizons': [5, 10, 20, 30],
            'confidence_intervals': [0.95]
        },
        'statistical': {
            'test_type': 'paired_t',
            'alpha': 0.05,
            'bootstrap_samples': 1000
        }
    }
}

metrics_calc = TrajectoryMetrics(eval_config)
stat_test = StatisticalTesting(eval_config)

# Model results storage
model_results = {}
model_predictions = {}

# 1. Evaluate Constant Velocity baseline
print("\\n=== Evaluating Constant Velocity Baseline ===")
with open('$OUTPUT_DIR/models/cv_baseline.pkl', 'rb') as f:
    cv_model = pickle.load(f)

start_time = time.time()
cv_pred = cv_model.predict(X_test)
cv_inference_time = time.time() - start_time

cv_results = metrics_calc.evaluate_model(cv_pred, y_test)
cv_results['inference_time'] = cv_inference_time

model_results['Constant Velocity'] = cv_results
model_predictions['Constant Velocity'] = cv_pred

print("CV Results:")
for metric, value in cv_results['overall_metrics'].items():
    print(f"  {metric.upper()}: {value:.4f}")

# 2. Evaluate LSTM baseline
print("\\n=== Evaluating LSTM Baseline ===")
with open('$CONFIG_DIR/model_lstm.yaml', 'r') as f:
    lstm_config = yaml.safe_load(f)

lstm_model = create_lstm_model(lstm_config, 'basic')
lstm_checkpoint = torch.load('$OUTPUT_DIR/models/lstm_baseline_best.pth', map_location=device)
lstm_model.load_state_dict(lstm_checkpoint)
lstm_model = lstm_model.to(device)
lstm_model.eval()

# LSTM inference
X_test_tensor = torch.FloatTensor(X_test).to(device)
lstm_predictions = []

start_time = time.time()
with torch.no_grad():
    batch_size = 64
    for i in range(0, len(X_test_tensor), batch_size):
        batch = X_test_tensor[i:i+batch_size]
        pred = lstm_model(batch)
        lstm_predictions.append(pred.cpu().numpy())

lstm_inference_time = time.time() - start_time
lstm_pred = np.concatenate(lstm_predictions, axis=0)

lstm_results = metrics_calc.evaluate_model(lstm_pred, y_test)
lstm_results['inference_time'] = lstm_inference_time

model_results['LSTM'] = lstm_results
model_predictions['LSTM'] = lstm_pred

print("LSTM Results:")
for metric, value in lstm_results['overall_metrics'].items():
    print(f"  {metric.upper()}: {value:.4f}")

# 3. Evaluate Transformer
print("\\n=== Evaluating Transformer ===")
with open('$CONFIG_DIR/model_transformer.yaml', 'r') as f:
    transformer_config = yaml.safe_load(f)

transformer_model = create_transformer_model(transformer_config, 'encoder')
transformer_checkpoint = torch.load('$OUTPUT_DIR/models/transformer_best.pth', map_location=device)
transformer_model.load_state_dict(transformer_checkpoint['model_state_dict'])
transformer_model = transformer_model.to(device)
transformer_model.eval()

# Transformer inference
transformer_predictions = []

start_time = time.time()
with torch.no_grad():
    batch_size = 64
    for i in range(0, len(X_test_tensor), batch_size):
        batch = X_test_tensor[i:i+batch_size]
        pred = transformer_model(batch)
        transformer_predictions.append(pred.cpu().numpy())

transformer_inference_time = time.time() - start_time
transformer_pred = np.concatenate(transformer_predictions, axis=0)

transformer_results = metrics_calc.evaluate_model(transformer_pred, y_test)
transformer_results['inference_time'] = transformer_inference_time

model_results['Transformer'] = transformer_results
model_predictions['Transformer'] = transformer_pred

print("Transformer Results:")
for metric, value in transformer_results['overall_metrics'].items():
    print(f"  {metric.upper()}: {value:.4f}")

# 4. Evaluate new models if available
if NEW_MODELS_IMPORTED and '$NEW_MODELS_AVAILABLE' == 'true':
    # GNN Model
    print("\\n=== Evaluating GNN Model ===")
    try:
        with open('$CONFIG_DIR/model_gnn.yaml', 'r') as f:
            gnn_config = yaml.safe_load(f)

        gnn_model = create_gnn_model(gnn_config, 'basic')
        gnn_checkpoint = torch.load('$OUTPUT_DIR/models/gnn_best.pth', map_location=device)
        gnn_model.load_state_dict(gnn_checkpoint['model_state_dict'])
        gnn_model = gnn_model.to(device)
        gnn_model.eval()

        gnn_predictions = []
        start_time = time.time()
        with torch.no_grad():
            batch_size = 32  # Smaller batch for GNN
            for i in range(0, len(X_test_tensor), batch_size):
                batch = X_test_tensor[i:i+batch_size]
                pred = gnn_model(batch)
                gnn_predictions.append(pred.cpu().numpy())

        gnn_inference_time = time.time() - start_time
        gnn_pred = np.concatenate(gnn_predictions, axis=0)

        gnn_results = metrics_calc.evaluate_model(gnn_pred, y_test)
        gnn_results['inference_time'] = gnn_inference_time

        model_results['GNN'] = gnn_results
        model_predictions['GNN'] = gnn_pred

        print("GNN Results:")
        for metric, value in gnn_results['overall_metrics'].items():
            print(f"  {metric.upper()}: {value:.4f}")

    except Exception as e:
        print(f"Error evaluating GNN: {e}")

    # STT Model
    print("\\n=== Evaluating STT Model ===")
    try:
        with open('$CONFIG_DIR/model_stt.yaml', 'r') as f:
            stt_config = yaml.safe_load(f)

        stt_model = create_stt_model(stt_config, 'basic')
        stt_checkpoint = torch.load('$OUTPUT_DIR/models/stt_best.pth', map_location=device)
        stt_model.load_state_dict(stt_checkpoint['model_state_dict'])
        stt_model = stt_model.to(device)
        stt_model.eval()

        stt_predictions = []
        start_time = time.time()
        with torch.no_grad():
            batch_size = 48
            for i in range(0, len(X_test_tensor), batch_size):
                batch = X_test_tensor[i:i+batch_size]
                pred = stt_model(batch)
                stt_predictions.append(pred.cpu().numpy())

        stt_inference_time = time.time() - start_time
        stt_pred = np.concatenate(stt_predictions, axis=0)

        stt_results = metrics_calc.evaluate_model(stt_pred, y_test)
        stt_results['inference_time'] = stt_inference_time

        model_results['STT'] = stt_results
        model_predictions['STT'] = stt_pred

        print("STT Results:")
        for metric, value in stt_results['overall_metrics'].items():
            print(f"  {metric.upper()}: {value:.4f}")

    except Exception as e:
        print(f"Error evaluating STT: {e}")

    # PINN Model
    print("\\n=== Evaluating PINN Model ===")
    try:
        with open('$CONFIG_DIR/model_pinn.yaml', 'r') as f:
            pinn_config = yaml.safe_load(f)

        pinn_model = create_pinn_model(pinn_config, 'basic')
        pinn_checkpoint = torch.load('$OUTPUT_DIR/models/pinn_best.pth', map_location=device)
        pinn_model.load_state_dict(pinn_checkpoint['model_state_dict'])
        pinn_model = pinn_model.to(device)
        pinn_model.eval()

        pinn_predictions = []
        start_time = time.time()
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(X_test_tensor), batch_size):
                batch = X_test_tensor[i:i+batch_size]
                pred = pinn_model(batch, return_physics_terms=False)  # Standard inference
                pinn_predictions.append(pred.cpu().numpy())

        pinn_inference_time = time.time() - start_time
        pinn_pred = np.concatenate(pinn_predictions, axis=0)

        pinn_results = metrics_calc.evaluate_model(pinn_pred, y_test)
        pinn_results['inference_time'] = pinn_inference_time

        model_results['PINN'] = pinn_results
        model_predictions['PINN'] = pinn_pred

        print("PINN Results:")
        for metric, value in pinn_results['overall_metrics'].items():
            print(f"  {metric.upper()}: {value:.4f}")

    except Exception as e:
        print(f"Error evaluating PINN: {e}")

# Statistical significance testing
print("\\n=== Statistical Significance Testing ===")

# Compare LSTM vs CV
lstm_vs_cv = stat_test.compare_models(
    {'ade': np.array([metrics_calc.calculate_ade(lstm_pred[i:i+1], y_test[i:i+1]) for i in range(len(y_test))])},
    {'ade': np.array([metrics_calc.calculate_ade(cv_pred[i:i+1], y_test[i:i+1]) for i in range(len(y_test))])},
    ("LSTM", "Constant Velocity")
)

# Compare Transformer vs LSTM
transformer_vs_lstm = stat_test.compare_models(
    {'ade': np.array([metrics_calc.calculate_ade(transformer_pred[i:i+1], y_test[i:i+1]) for i in range(len(y_test))])},
    {'ade': np.array([metrics_calc.calculate_ade(lstm_pred[i:i+1], y_test[i:i+1]) for i in range(len(y_test))])},
    ("Transformer", "LSTM")
)

# Compare Transformer vs CV
transformer_vs_cv = stat_test.compare_models(
    {'ade': np.array([metrics_calc.calculate_ade(transformer_pred[i:i+1], y_test[i:i+1]) for i in range(len(y_test))])},
    {'ade': np.array([metrics_calc.calculate_ade(cv_pred[i:i+1], y_test[i:i+1]) for i in range(len(y_test))])},
    ("Transformer", "Constant Velocity")
)

print("LSTM vs CV:", "Significant" if lstm_vs_cv['metric_comparisons']['ade']['significant'] else "Not significant")
print("Transformer vs LSTM:", "Significant" if transformer_vs_lstm['metric_comparisons']['ade']['significant'] else "Not significant")
print("Transformer vs CV:", "Significant" if transformer_vs_cv['metric_comparisons']['ade']['significant'] else "Not significant")

# Save all results
all_results = {
    'model_results': model_results,
    'statistical_comparisons': {
        'lstm_vs_cv': lstm_vs_cv,
        'transformer_vs_lstm': transformer_vs_lstm,
        'transformer_vs_cv': transformer_vs_cv
    },
    'test_set_size': len(test_samples),
    'evaluation_config': eval_config
}

with open('$OUTPUT_DIR/models/comprehensive_evaluation.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print("\\nSaved comprehensive evaluation results")

# Create comparison plots
print("Creating comparison visualizations...")

# 1. Overall metrics comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
metrics = ['ade', 'fde', 'rmse', 'mae']
models = list(model_results.keys())  # Dynamic model list

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    values = [model_results[model]['overall_metrics'][metric] for model in models]
    # Use a colormap for dynamic colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, values, color=colors)
    ax.set_title(f'{metric.upper()} Comparison')
    ax.set_ylabel(f'{metric.upper()}')
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels if many models
    if len(models) > 4:
        ax.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('$OUTPUT_DIR/figures/model_comparison_overall.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Horizon-based performance
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
horizons = [5, 10, 20, 30]

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]

    for model in models:
        horizon_values = []
        for horizon in horizons:
            if horizon in model_results[model]['horizon_metrics']:
                horizon_values.append(model_results[model]['horizon_metrics'][horizon][metric])
            else:
                horizon_values.append(np.nan)

        ax.plot(horizons, horizon_values, marker='o', label=model, linewidth=2, markersize=6)

    ax.set_title(f'{metric.upper()} by Prediction Horizon')
    ax.set_xlabel('Prediction Horizon (minutes)')
    ax.set_ylabel(f'{metric.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('$OUTPUT_DIR/figures/model_comparison_horizons.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Inference time comparison
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
inference_times = [model_results[model]['inference_time'] for model in models]
colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
bars = ax.bar(models, inference_times, color=colors)
ax.set_title('Inference Time Comparison')
ax.set_ylabel('Inference Time (seconds)')
ax.grid(True, alpha=0.3)

if len(models) > 4:
    ax.tick_params(axis='x', rotation=45)

for bar, time_val in zip(bars, inference_times):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
            f'{time_val:.2f}s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('$OUTPUT_DIR/figures/inference_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved comparison visualizations")

# Generate evaluation report
report_text = create_evaluation_report(
    transformer_results,
    {},  # No avoidance results yet
    transformer_vs_lstm
)

with open('$OUTPUT_DIR/reports/evaluation_report.md', 'w') as f:
    f.write(report_text)

print("Saved evaluation report")

# Print summary
print("\\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)

print("\\nOverall Performance (ADE):")
for model in models:
    ade = model_results[model]['overall_metrics']['ade']
    time_val = model_results[model]['inference_time']
    print(f"  {model:18s}: {ade:.4f} ({time_val:.2f}s)")

print("\\nImprovement vs Baselines:")
lstm_ade = model_results['LSTM']['overall_metrics']['ade']
cv_ade = model_results['Constant Velocity']['overall_metrics']['ade']
transformer_ade = model_results['Transformer']['overall_metrics']['ade']

lstm_improvement = (cv_ade - lstm_ade) / cv_ade * 100
transformer_improvement_vs_lstm = (lstm_ade - transformer_ade) / lstm_ade * 100
transformer_improvement_vs_cv = (cv_ade - transformer_ade) / cv_ade * 100

print(f"  LSTM vs CV:        {lstm_improvement:+.1f}%")
print(f"  Transformer vs LSTM: {transformer_improvement_vs_lstm:+.1f}%")
print(f"  Transformer vs CV:   {transformer_improvement_vs_cv:+.1f}%")

print("\\nStatistical Significance:")
print(f"  LSTM vs CV:        {'✓' if lstm_vs_cv['metric_comparisons']['ade']['significant'] else '✗'}")
print(f"  Transformer vs LSTM: {'✓' if transformer_vs_lstm['metric_comparisons']['ade']['significant'] else '✗'}")
print(f"  Transformer vs CV:   {'✓' if transformer_vs_cv['metric_comparisons']['ade']['significant'] else '✗'}")

# Check if target improvements are met
target_improvement = 15.0  # 15% improvement target
meets_target = transformer_improvement_vs_lstm >= target_improvement

print(f"\\nTarget Achievement:")
print(f"  Target: ≥{target_improvement}% improvement over LSTM")
print(f"  Actual: {transformer_improvement_vs_lstm:.1f}%")
print(f"  Status: {'✓ ACHIEVED' if meets_target else '✗ NOT ACHIEVED'}")

print("\\n" + "="*60)
EOF

echo "=== Model Evaluation Complete ==="
echo "Results saved to: $OUTPUT_DIR/models/comprehensive_evaluation.json"
echo "Figures saved to: $OUTPUT_DIR/figures/"
echo "Report saved to: $OUTPUT_DIR/reports/evaluation_report.md"
echo ""
echo "Next steps:"
echo "1. Run collision avoidance simulation: bash scripts/40_sim_avoidance.sh"
