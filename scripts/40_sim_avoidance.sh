#!/bin/bash
# Script to run collision avoidance simulation and evaluation

set -e  # Exit on any error

echo "=== Collision Avoidance Simulation ==="
echo "Starting collision avoidance simulation and evaluation..."

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
CONFIG_DIR="$PROJECT_ROOT/configs"
OUTPUT_DIR="$PROJECT_ROOT/outputs"

echo "Project root: $PROJECT_ROOT"

# Create reports directory
mkdir -p "$OUTPUT_DIR/reports"

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

# Run collision avoidance simulation
echo "Running collision avoidance simulation..."
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

from risk.cpa_tcpa import VesselState, CPATCPACalculator, MultiVesselRiskAssessor
from avoidance.rule_search import AvoidanceOptimizer, MultiVesselAvoidanceCoordinator
from models.transformer import create_transformer_model
from eval.metrics import CollisionAvoidanceMetrics

# Load configuration
with open('$CONFIG_DIR/sim.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Configuration loaded")

# Load best transformer model for trajectory prediction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open('$CONFIG_DIR/model_transformer.yaml', 'r') as f:
    transformer_config = yaml.safe_load(f)

transformer_model = create_transformer_model(transformer_config, 'encoder')
transformer_checkpoint = torch.load('$OUTPUT_DIR/models/transformer_best.pth', map_location=device)
transformer_model.load_state_dict(transformer_checkpoint['model_state_dict'])
transformer_model = transformer_model.to(device)
transformer_model.eval()

print("Transformer model loaded")

# Initialize collision avoidance components
risk_assessor = MultiVesselRiskAssessor(config)
avoidance_coordinator = MultiVesselAvoidanceCoordinator(config)
ca_metrics = CollisionAvoidanceMetrics(config)

print("Collision avoidance components initialized")

def create_encounter_scenario(scenario_type="crossing", distance_nmi=1.0):
    """Create a specific encounter scenario."""
    np.random.seed(42)  # For reproducibility

    if scenario_type == "crossing":
        # Two vessels on crossing courses
        vessel_a = VesselState(
            mmsi=123456789, timestamp=0, lat=40.7128, lon=-74.0060,
            x=0, y=0, sog=15.0, cog=90.0  # Eastbound
        )

        vessel_b = VesselState(
            mmsi=987654321, timestamp=0, lat=40.7128 + distance_nmi/60, lon=-74.0060,
            x=0, y=distance_nmi * 1852, sog=12.0, cog=180.0  # Southbound
        )

    elif scenario_type == "head_on":
        # Two vessels on opposite courses
        vessel_a = VesselState(
            mmsi=123456789, timestamp=0, lat=40.7128, lon=-74.0060,
            x=0, y=0, sog=15.0, cog=90.0  # Eastbound
        )

        vessel_b = VesselState(
            mmsi=987654321, timestamp=0, lat=40.7128, lon=-74.0060 + distance_nmi/60,
            x=distance_nmi * 1852, y=0, sog=12.0, cog=270.0  # Westbound
        )

    elif scenario_type == "overtaking":
        # Faster vessel overtaking slower one
        vessel_a = VesselState(
            mmsi=123456789, timestamp=0, lat=40.7128, lon=-74.0060,
            x=0, y=0, sog=10.0, cog=90.0  # Slower vessel
        )

        vessel_b = VesselState(
            mmsi=987654321, timestamp=0, lat=40.7128, lon=-74.0060 - distance_nmi/120,
            x=-distance_nmi * 1852 / 2, y=100, sog=18.0, cog=90.0  # Faster vessel, slightly offset
        )

    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    return [vessel_a, vessel_b]

# Generate test scenarios
print("Generating test scenarios...")
scenario_types = ["crossing", "head_on", "overtaking"]
distances = [0.5, 1.0, 1.5, 2.0]  # nautical miles
n_monte_carlo = config['simulation']['scenarios']['n_monte_carlo']

all_scenarios = []
scenario_labels = []

for scenario_type in scenario_types:
    for distance in distances:
        for run in range(n_monte_carlo // (len(scenario_types) * len(distances))):
            vessels = create_encounter_scenario(scenario_type, distance)
            all_scenarios.append(vessels)
            scenario_labels.append(f"{scenario_type}_{distance}nmi_{run}")

print(f"Generated {len(all_scenarios)} scenarios")

# Run simulations
print("Running collision avoidance simulations...")

simulation_results = []
avoidance_results = []

for i, (vessels, label) in enumerate(zip(all_scenarios, scenario_labels)):
    if i % 20 == 0:
        print(f"Processing scenario {i+1}/{len(all_scenarios)}: {label}")

    try:
        # Run simulation with avoidance
        sim_result = avoidance_coordinator.simulate_avoidance_scenario(vessels, time_steps=30)
        simulation_results.append(sim_result)

        # Extract avoidance actions from simulation log
        scenario_actions = []
        for step_log in sim_result['simulation_log']:
            if step_log['actions']:
                scenario_actions.extend(step_log['actions'].values())

        if scenario_actions:
            avoidance_results.extend(scenario_actions)

    except Exception as e:
        print(f"Warning: Simulation failed for scenario {label}: {e}")
        continue

print(f"Completed {len(simulation_results)} simulations")

# Evaluate collision avoidance performance
print("Evaluating collision avoidance performance...")

ca_evaluation = ca_metrics.evaluate_scenario_batch(simulation_results)
action_analysis = ca_metrics.analyze_avoidance_actions(avoidance_results)

print("Collision Avoidance Results:")
print(f"  Total scenarios: {ca_evaluation['n_scenarios']}")
print(f"  Collision rate: {ca_evaluation['collision_rate']:.1%}")
print(f"  Near miss rate: {ca_evaluation['near_miss_rate']:.1%}")
print(f"  Safe encounter rate: {ca_evaluation['safe_encounter_rate']:.1%}")
print(f"  Average CPA: {ca_evaluation['cpa_statistics']['mean']:.3f} nmi")

# Check success criteria
success_criteria = ca_evaluation['success_criteria']
print(f"\\nSuccess Criteria:")
print(f"  Collision rate ≤ 2%: {'✓' if success_criteria['collision_rate_ok'] else '✗'}")
print(f"  Average CPA ≥ 0.2 nmi: {'✓' if success_criteria['avg_cpa_ok'] else '✗'}")
print(f"  Overall success: {'✓' if success_criteria['overall_success'] else '✗'}")

# Save results
results_data = {
    'collision_avoidance_evaluation': ca_evaluation,
    'avoidance_action_analysis': action_analysis,
    'scenario_details': {
        'total_scenarios': len(all_scenarios),
        'scenario_types': scenario_types,
        'distances_tested': distances,
        'monte_carlo_runs': n_monte_carlo
    },
    'simulation_results': simulation_results[:10],  # Save first 10 for analysis
    'config': config
}

with open('$OUTPUT_DIR/models/collision_avoidance_results.json', 'w') as f:
    json.dump(results_data, f, indent=2, default=str)

print("\\nSaved collision avoidance results")

# Create visualization
print("Creating collision avoidance visualizations...")

# 1. CPA distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# CPA histogram
ax = axes[0, 0]
cpas = [result['min_cpa_achieved'] for result in simulation_results]
ax.hist(cpas, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax.axvline(x=0.1, color='red', linestyle='--', label='Collision threshold')
ax.axvline(x=0.2, color='orange', linestyle='--', label='Near miss threshold')
ax.set_xlabel('Minimum CPA (nautical miles)')
ax.set_ylabel('Frequency')
ax.set_title('CPA Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Success rate by scenario type
ax = axes[0, 1]
scenario_success = {}
for scenario_type in scenario_types:
    scenario_results = [r for r, l in zip(simulation_results, scenario_labels) if scenario_type in l]
    success_rate = sum(1 for r in scenario_results if not r['collision_occurred']) / len(scenario_results)
    scenario_success[scenario_type] = success_rate

bars = ax.bar(scenario_success.keys(), [v * 100 for v in scenario_success.values()],
              color=['lightcoral', 'lightgreen', 'lightsalmon'])
ax.set_ylabel('Success Rate (%)')
ax.set_title('Success Rate by Scenario Type')
ax.grid(True, alpha=0.3)

for bar, value in zip(bars, scenario_success.values()):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{value:.1%}', ha='center', va='bottom')

# Action distribution
ax = axes[1, 0]
if action_analysis and 'action_distribution' in action_analysis:
    actions = list(action_analysis['action_distribution'].keys())
    counts = list(action_analysis['action_distribution'].values())
    ax.pie(counts, labels=actions, autopct='%1.1f%%', startangle=90)
    ax.set_title('Avoidance Action Distribution')

# CPA improvement
ax = axes[1, 1]
if avoidance_results:
    improvements = [r['improvement'] for r in avoidance_results]
    ax.hist(improvements, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.set_xlabel('CPA Improvement (nautical miles)')
    ax.set_ylabel('Frequency')
    ax.set_title('CPA Improvement Distribution')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('$OUTPUT_DIR/figures/collision_avoidance_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Sample scenario visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot a sample scenario
if simulation_results:
    sample_sim = simulation_results[0]

    for step_log in sample_sim['simulation_log'][::5]:  # Every 5th step
        vessels = step_log['vessels']

        for vessel in vessels:
            ax.scatter(vessel.x, vessel.y, s=50, alpha=0.6,
                      label=f'MMSI {vessel.mmsi}' if step_log['step'] == 0 else "")

        # Draw risk zones
        if step_log['risks']:
            for risk in step_log['risks']:
                if risk.cpa_distance < 0.5:  # High risk
                    # Draw circle around vessels
                    circle = plt.Circle((vessels[0].x, vessels[0].y), 500,
                                      fill=False, color='red', alpha=0.3)
                    ax.add_patch(circle)

ax.set_xlabel('X Position (meters)')
ax.set_ylabel('Y Position (meters)')
ax.set_title('Sample Collision Avoidance Scenario')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.tight_layout()
plt.savefig('$OUTPUT_DIR/figures/sample_avoidance_scenario.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved collision avoidance visualizations")

# Generate final report
report_lines = []
report_lines.append("# Ship Collision Avoidance Evaluation Report")
report_lines.append("")
report_lines.append("## Executive Summary")
report_lines.append("")
report_lines.append(f"This report presents the evaluation results of the ship collision avoidance system")
report_lines.append(f"based on {len(simulation_results)} Monte Carlo simulation scenarios.")
report_lines.append("")
report_lines.append("## Key Results")
report_lines.append("")
report_lines.append(f"- **Collision Rate**: {ca_evaluation['collision_rate']:.1%}")
report_lines.append(f"- **Near Miss Rate**: {ca_evaluation['near_miss_rate']:.1%}")
report_lines.append(f"- **Safe Encounter Rate**: {ca_evaluation['safe_encounter_rate']:.1%}")
report_lines.append(f"- **Average CPA**: {ca_evaluation['cpa_statistics']['mean']:.3f} nautical miles")
report_lines.append(f"- **Minimum CPA**: {ca_evaluation['cpa_statistics']['min']:.3f} nautical miles")
report_lines.append("")
report_lines.append("## Success Criteria Assessment")
report_lines.append("")
report_lines.append(f"| Criterion | Target | Actual | Status |")
report_lines.append(f"|-----------|--------|--------|--------|")
report_lines.append(f"| Collision Rate | ≤ 2% | {ca_evaluation['collision_rate']:.1%} | {'✓' if success_criteria['collision_rate_ok'] else '✗'} |")
report_lines.append(f"| Average CPA | ≥ 0.2 nmi | {ca_evaluation['cpa_statistics']['mean']:.3f} nmi | {'✓' if success_criteria['avg_cpa_ok'] else '✗'} |")
report_lines.append("")
report_lines.append(f"**Overall Success**: {'✓ ACHIEVED' if success_criteria['overall_success'] else '✗ NOT ACHIEVED'}")
report_lines.append("")

if action_analysis:
    report_lines.append("## Avoidance Action Analysis")
    report_lines.append("")
    report_lines.append(f"- **Total Actions Taken**: {len(avoidance_results)}")
    report_lines.append(f"- **Overall Success Rate**: {action_analysis.get('overall_success_rate', 0):.1%}")
    report_lines.append(f"- **Average CPA Improvement**: {action_analysis['improvement_statistics']['mean']:.3f} nmi")
    report_lines.append("")

report_text = "\\n".join(report_lines)

with open('$OUTPUT_DIR/reports/collision_avoidance_report.md', 'w') as f:
    f.write(report_text)

print("Saved collision avoidance report")

print("\\n" + "="*60)
print("COLLISION AVOIDANCE EVALUATION SUMMARY")
print("="*60)
print(f"Total Scenarios: {len(simulation_results)}")
print(f"Collision Rate: {ca_evaluation['collision_rate']:.1%}")
print(f"Average CPA: {ca_evaluation['cpa_statistics']['mean']:.3f} nmi")
print(f"Success Criteria: {'✓ ACHIEVED' if success_criteria['overall_success'] else '✗ NOT ACHIEVED'}")
print("="*60)
EOF

echo "=== Collision Avoidance Simulation Complete ==="
echo "Results saved to: $OUTPUT_DIR/models/collision_avoidance_results.json"
echo "Figures saved to: $OUTPUT_DIR/figures/"
echo "Report saved to: $OUTPUT_DIR/reports/collision_avoidance_report.md"
echo ""
echo "All simulations and evaluations completed successfully!"
