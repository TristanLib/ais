"""
Rule-based collision avoidance with optimization search.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import copy

from ..risk.cpa_tcpa import (
    CPATCPACalculator, VesselState, RiskAssessment, RiskLevel
)

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of avoidance actions."""
    MAINTAIN = "maintain"
    TURN_PORT = "turn_port"
    TURN_STARBOARD = "turn_starboard"
    SPEED_UP = "speed_up"
    SLOW_DOWN = "slow_down"
    COMBINED = "combined"


@dataclass
class AvoidanceAction:
    """Avoidance action specification."""
    action_type: ActionType
    heading_change: float  # degrees (positive = starboard, negative = port)
    speed_change: float    # relative change (e.g., -0.1 = 10% reduction)
    duration: float        # action duration in minutes
    description: str

    def apply_to_vessel(self, vessel: VesselState) -> VesselState:
        """Apply action to vessel state."""
        new_vessel = copy.deepcopy(vessel)

        # Apply heading change
        if abs(self.heading_change) > 0.1:
            new_vessel.cog = (vessel.cog + self.heading_change) % 360

        # Apply speed change
        if abs(self.speed_change) > 0.01:
            new_vessel.sog = max(0.1, vessel.sog * (1 + self.speed_change))

        return new_vessel


@dataclass
class AvoidanceResult:
    """Result of avoidance optimization."""
    original_risk: RiskAssessment
    best_action: AvoidanceAction
    new_risk: RiskAssessment
    improvement: float  # CPA improvement in nautical miles
    cost: float         # Action cost
    success: bool       # Whether collision was successfully avoided
    all_candidates: List[Tuple[AvoidanceAction, RiskAssessment, float]]


class COLREGSRules:
    """Implementation of COLREGs rules for collision avoidance."""

    def __init__(self, config: Dict):
        """Initialize COLREGs rules."""
        self.config = config
        colregs_config = config.get('simulation', {}).get('avoidance', {}).get('colregs', {})

        self.enable = colregs_config.get('enable', True)
        self.give_way_bias = colregs_config.get('give_way_bias', 1.5)
        self.starboard_preference = colregs_config.get('starboard_preference', 1.2)

    def determine_give_way_vessel(self, vessel_a: VesselState, vessel_b: VesselState,
                                encounter_type: str) -> str:
        """
        Determine which vessel should give way according to COLREGs.

        Args:
            vessel_a: Target vessel
            vessel_b: Other vessel
            encounter_type: Type of encounter

        Returns:
            'A' if vessel_a gives way, 'B' if vessel_b gives way, 'both' for head-on
        """
        if encounter_type == 'head-on':
            return 'both'  # Both vessels turn starboard

        elif encounter_type == 'overtaking':
            # Overtaking vessel gives way
            if vessel_b.sog > vessel_a.sog:
                return 'B'  # B is overtaking A
            else:
                return 'A'  # A is overtaking B

        elif encounter_type == 'crossing':
            # Vessel with other on starboard side gives way
            # Calculate relative bearing
            dx = vessel_b.x - vessel_a.x
            dy = vessel_b.y - vessel_a.y
            bearing_a_to_b = np.rad2deg(np.arctan2(dx, dy)) % 360
            relative_bearing = (bearing_a_to_b - vessel_a.cog) % 360

            if relative_bearing < 180:  # B is on A's starboard side
                return 'A'
            else:  # B is on A's port side
                return 'B'

        return 'unknown'

    def get_preferred_actions(self, give_way_vessel: str, encounter_type: str) -> List[ActionType]:
        """Get preferred actions based on COLREGs."""
        if not self.enable:
            return [ActionType.TURN_STARBOARD, ActionType.TURN_PORT, ActionType.SLOW_DOWN]

        if encounter_type == 'head-on':
            return [ActionType.TURN_STARBOARD]  # Both turn starboard

        elif encounter_type == 'crossing':
            if give_way_vessel == 'A':
                # Give way vessel should turn starboard or slow down
                return [ActionType.TURN_STARBOARD, ActionType.SLOW_DOWN, ActionType.TURN_PORT]
            else:
                # Stand-on vessel maintains course
                return [ActionType.MAINTAIN]

        elif encounter_type == 'overtaking':
            # Overtaking vessel should alter course
            return [ActionType.TURN_PORT, ActionType.TURN_STARBOARD, ActionType.SLOW_DOWN]

        return [ActionType.TURN_STARBOARD, ActionType.SLOW_DOWN, ActionType.TURN_PORT]


class AvoidanceOptimizer:
    """Collision avoidance optimizer using rule-based search."""

    def __init__(self, config: Dict):
        """Initialize avoidance optimizer."""
        self.config = config
        avoidance_config = config.get('simulation', {}).get('avoidance', {})

        # Action space
        self.heading_changes = avoidance_config.get('heading_changes', [-20, -10, 0, 10, 20])
        self.speed_changes = avoidance_config.get('speed_changes', [-0.2, -0.1, 0, 0.1])

        # Constraints
        self.max_heading_rate = avoidance_config.get('max_heading_rate', 5)  # deg/min
        self.max_speed_change = avoidance_config.get('max_speed_change', 0.3)
        self.min_speed_ratio = avoidance_config.get('min_speed_ratio', 0.3)

        # Cost function weights
        weights = avoidance_config.get('weights', {})
        self.weight_safety = weights.get('safety', 10.0)
        self.weight_deviation = weights.get('deviation', 1.0)
        self.weight_energy = weights.get('energy', 0.5)
        self.weight_comfort = weights.get('comfort', 0.3)

        # Components
        self.risk_calculator = CPATCPACalculator(config)
        self.colregs = COLREGSRules(config)

    def generate_candidate_actions(self, vessel: VesselState, encounter_type: str,
                                 give_way_vessel: str) -> List[AvoidanceAction]:
        """Generate candidate avoidance actions."""
        candidates = []

        # Get preferred actions from COLREGs
        preferred_actions = self.colregs.get_preferred_actions(give_way_vessel, encounter_type)

        # Generate actions based on action space
        for heading_change in self.heading_changes:
            for speed_change in self.speed_changes:

                # Skip no-action if we need to avoid
                if heading_change == 0 and speed_change == 0:
                    action_type = ActionType.MAINTAIN
                elif heading_change > 0:
                    action_type = ActionType.TURN_STARBOARD
                elif heading_change < 0:
                    action_type = ActionType.TURN_PORT
                elif speed_change > 0:
                    action_type = ActionType.SPEED_UP
                elif speed_change < 0:
                    action_type = ActionType.SLOW_DOWN
                else:
                    continue

                # Check if action is preferred by COLREGs
                if self.colregs.enable and action_type not in preferred_actions:
                    continue

                # Check constraints
                if not self._check_action_constraints(vessel, heading_change, speed_change):
                    continue

                # Create action
                description = self._create_action_description(heading_change, speed_change)
                action = AvoidanceAction(
                    action_type=action_type,
                    heading_change=heading_change,
                    speed_change=speed_change,
                    duration=5.0,  # 5 minutes default
                    description=description
                )

                candidates.append(action)

        return candidates

    def _check_action_constraints(self, vessel: VesselState,
                                heading_change: float, speed_change: float) -> bool:
        """Check if action satisfies constraints."""
        # Heading rate constraint
        if abs(heading_change) > self.max_heading_rate * 5:  # 5-minute action
            return False

        # Speed change constraint
        if abs(speed_change) > self.max_speed_change:
            return False

        # Minimum speed constraint
        new_speed = vessel.sog * (1 + speed_change)
        min_speed = vessel.sog * self.min_speed_ratio
        if new_speed < min_speed:
            return False

        return True

    def _create_action_description(self, heading_change: float, speed_change: float) -> str:
        """Create human-readable action description."""
        parts = []

        if abs(heading_change) > 0.1:
            direction = "starboard" if heading_change > 0 else "port"
            parts.append(f"turn {abs(heading_change):.0f}° {direction}")

        if abs(speed_change) > 0.01:
            if speed_change > 0:
                parts.append(f"increase speed {speed_change*100:.0f}%")
            else:
                parts.append(f"reduce speed {abs(speed_change)*100:.0f}%")

        if not parts:
            return "maintain course and speed"

        return " and ".join(parts)

    def evaluate_action(self, target_vessel: VesselState, other_vessel: VesselState,
                       action: AvoidanceAction, original_risk: RiskAssessment) -> Tuple[RiskAssessment, float]:
        """
        Evaluate an avoidance action.

        Args:
            target_vessel: Target vessel state
            other_vessel: Other vessel state
            action: Avoidance action to evaluate
            original_risk: Original risk assessment

        Returns:
            Tuple of (new_risk_assessment, action_cost)
        """
        # Apply action to target vessel
        modified_vessel = action.apply_to_vessel(target_vessel)

        # Calculate new risk
        new_risk = self.risk_calculator.assess_collision_risk(modified_vessel, other_vessel)

        # Calculate action cost
        cost = self._calculate_action_cost(action, original_risk, new_risk)

        return new_risk, cost

    def _calculate_action_cost(self, action: AvoidanceAction,
                             original_risk: RiskAssessment,
                             new_risk: RiskAssessment) -> float:
        """Calculate cost of an avoidance action."""
        # Safety improvement (negative cost = benefit)
        cpa_improvement = new_risk.cpa_distance - original_risk.cpa_distance
        safety_cost = -self.weight_safety * cpa_improvement

        # Deviation cost
        deviation_cost = (self.weight_deviation *
                         (abs(action.heading_change) / 20.0 + abs(action.speed_change)))

        # Energy cost (approximation)
        energy_cost = self.weight_energy * abs(action.speed_change)

        # Comfort cost (sudden maneuvers)
        comfort_cost = self.weight_comfort * (abs(action.heading_change) / 10.0) ** 2

        total_cost = safety_cost + deviation_cost + energy_cost + comfort_cost

        return total_cost

    def optimize_avoidance(self, target_vessel: VesselState, other_vessel: VesselState,
                          risk_assessment: RiskAssessment) -> AvoidanceResult:
        """
        Optimize avoidance action for a collision risk scenario.

        Args:
            target_vessel: Target vessel state
            other_vessel: Other vessel state
            risk_assessment: Current risk assessment

        Returns:
            AvoidanceResult with optimal action
        """
        # Check if avoidance is needed
        if risk_assessment.risk_level in [RiskLevel.SAFE, RiskLevel.LOW]:
            maintain_action = AvoidanceAction(
                ActionType.MAINTAIN, 0, 0, 0, "maintain course and speed"
            )
            return AvoidanceResult(
                original_risk=risk_assessment,
                best_action=maintain_action,
                new_risk=risk_assessment,
                improvement=0.0,
                cost=0.0,
                success=True,
                all_candidates=[]
            )

        # Determine COLREGs situation
        encounter_type = risk_assessment.encounter_geometry.encounter_type
        give_way_vessel = self.colregs.determine_give_way_vessel(
            target_vessel, other_vessel, encounter_type
        )

        # Generate candidate actions
        candidates = self.generate_candidate_actions(target_vessel, encounter_type, give_way_vessel)

        if not candidates:
            logger.warning("No valid candidate actions generated")
            maintain_action = AvoidanceAction(
                ActionType.MAINTAIN, 0, 0, 0, "no valid actions available"
            )
            return AvoidanceResult(
                original_risk=risk_assessment,
                best_action=maintain_action,
                new_risk=risk_assessment,
                improvement=0.0,
                cost=float('inf'),
                success=False,
                all_candidates=[]
            )

        # Evaluate all candidates
        evaluated_candidates = []
        best_action = None
        best_cost = float('inf')
        best_risk = None

        for action in candidates:
            try:
                new_risk, cost = self.evaluate_action(
                    target_vessel, other_vessel, action, risk_assessment
                )
                evaluated_candidates.append((action, new_risk, cost))

                # Update best action
                if cost < best_cost:
                    best_cost = cost
                    best_action = action
                    best_risk = new_risk

            except Exception as e:
                logger.warning(f"Failed to evaluate action {action.description}: {e}")
                continue

        if best_action is None:
            logger.error("No actions could be evaluated successfully")
            maintain_action = AvoidanceAction(
                ActionType.MAINTAIN, 0, 0, 0, "evaluation failed"
            )
            return AvoidanceResult(
                original_risk=risk_assessment,
                best_action=maintain_action,
                new_risk=risk_assessment,
                improvement=0.0,
                cost=float('inf'),
                success=False,
                all_candidates=evaluated_candidates
            )

        # Calculate improvement
        improvement = best_risk.cpa_distance - risk_assessment.cpa_distance

        # Determine success
        success = (best_risk.risk_level in [RiskLevel.SAFE, RiskLevel.LOW] or
                  improvement > 0.1)  # At least 0.1 nmi improvement

        return AvoidanceResult(
            original_risk=risk_assessment,
            best_action=best_action,
            new_risk=best_risk,
            improvement=improvement,
            cost=best_cost,
            success=success,
            all_candidates=evaluated_candidates
        )


class MultiVesselAvoidanceCoordinator:
    """Coordinate avoidance actions for multiple vessels."""

    def __init__(self, config: Dict):
        """Initialize multi-vessel coordinator."""
        self.config = config
        self.optimizer = AvoidanceOptimizer(config)
        self.risk_calculator = CPATCPACalculator(config)

    def coordinate_avoidance(self, vessels: List[VesselState],
                           risk_assessments: List[RiskAssessment]) -> Dict[int, AvoidanceResult]:
        """
        Coordinate avoidance actions for multiple vessels.

        Args:
            vessels: List of vessel states
            risk_assessments: List of risk assessments

        Returns:
            Dictionary mapping MMSI to avoidance results
        """
        avoidance_results = {}

        # Sort risks by severity (most critical first)
        risk_order = {
            RiskLevel.CRITICAL: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 1,
            RiskLevel.SAFE: 0
        }

        sorted_risks = sorted(risk_assessments,
                            key=lambda r: (-risk_order[r.risk_level], r.tcpa))

        # Process each high-risk encounter
        for risk in sorted_risks:
            if risk.risk_level in [RiskLevel.SAFE, RiskLevel.LOW]:
                continue

            # Find vessels involved
            target_vessel = next((v for v in vessels if v.mmsi == risk.target_mmsi), None)
            other_vessel = next((v for v in vessels if v.mmsi == risk.other_mmsi), None)

            if target_vessel is None or other_vessel is None:
                logger.warning(f"Could not find vessels for risk assessment: "
                             f"{risk.target_mmsi} vs {risk.other_mmsi}")
                continue

            # Skip if either vessel already has an avoidance action
            if (risk.target_mmsi in avoidance_results or
                risk.other_mmsi in avoidance_results):
                continue

            # Optimize avoidance for target vessel
            result = self.optimizer.optimize_avoidance(target_vessel, other_vessel, risk)

            if result.success:
                avoidance_results[risk.target_mmsi] = result
                logger.info(f"Assigned avoidance action to vessel {risk.target_mmsi}: "
                          f"{result.best_action.description}")

        return avoidance_results

    def simulate_avoidance_scenario(self, vessels: List[VesselState],
                                  time_steps: int = 30) -> Dict:
        """
        Simulate avoidance scenario over multiple time steps.

        Args:
            vessels: Initial vessel states
            time_steps: Number of time steps to simulate

        Returns:
            Simulation results dictionary
        """
        # Initialize simulation state
        current_vessels = copy.deepcopy(vessels)
        simulation_log = []
        collision_occurred = False
        min_cpa_achieved = float('inf')

        for step in range(time_steps):
            # Assess current risks
            risk_assessments = []
            for i in range(len(current_vessels)):
                for j in range(i + 1, len(current_vessels)):
                    risk = self.risk_calculator.assess_collision_risk(
                        current_vessels[i], current_vessels[j]
                    )
                    risk_assessments.append(risk)

            # Check for collisions
            for risk in risk_assessments:
                if risk.cpa_distance < 0.05:  # 50 meters collision threshold
                    collision_occurred = True
                min_cpa_achieved = min(min_cpa_achieved, risk.cpa_distance)

            # Coordinate avoidance actions
            avoidance_results = self.coordinate_avoidance(current_vessels, risk_assessments)

            # Log current state
            step_log = {
                'step': step,
                'vessels': copy.deepcopy(current_vessels),
                'risks': risk_assessments,
                'actions': avoidance_results,
                'min_cpa': min([r.cpa_distance for r in risk_assessments]) if risk_assessments else float('inf')
            }
            simulation_log.append(step_log)

            # Update vessel states (simplified kinematic model)
            for i, vessel in enumerate(current_vessels):
                # Apply avoidance action if assigned
                if vessel.mmsi in avoidance_results:
                    action = avoidance_results[vessel.mmsi].best_action
                    current_vessels[i] = action.apply_to_vessel(vessel)

                # Update position (1 minute time step)
                dt_hours = 1.0 / 60.0  # 1 minute
                speed_ms = vessel.sog * 0.514444  # knots to m/s

                # Update position
                cog_rad = np.deg2rad(vessel.cog)
                dx = speed_ms * np.sin(cog_rad) * dt_hours * 3600  # meters
                dy = speed_ms * np.cos(cog_rad) * dt_hours * 3600  # meters

                current_vessels[i].x += dx
                current_vessels[i].y += dy
                current_vessels[i].timestamp += 60  # 1 minute

        return {
            'collision_occurred': collision_occurred,
            'min_cpa_achieved': min_cpa_achieved,
            'final_vessels': current_vessels,
            'simulation_log': simulation_log,
            'total_steps': time_steps
        }


if __name__ == "__main__":
    # Example usage
    config = {
        'simulation': {
            'avoidance': {
                'heading_changes': [-20, -10, 0, 10, 20],
                'speed_changes': [-0.2, -0.1, 0, 0.1],
                'weights': {
                    'safety': 10.0,
                    'deviation': 1.0,
                    'energy': 0.5,
                    'comfort': 0.3
                },
                'colregs': {
                    'enable': True,
                    'give_way_bias': 1.5,
                    'starboard_preference': 1.2
                }
            },
            'risk': {
                'safe_distance': 0.5,
                'time_horizon': 1800
            }
        }
    }

    # Test scenario
    vessel_a = VesselState(
        mmsi=123456789, timestamp=0, lat=40.7128, lon=-74.0060,
        x=0, y=0, sog=15.0, cog=90.0
    )

    vessel_b = VesselState(
        mmsi=987654321, timestamp=0, lat=40.7138, lon=-74.0050,
        x=1000, y=1000, sog=12.0, cog=270.0
    )

    # Test avoidance optimization
    optimizer = AvoidanceOptimizer(config)
    risk_calc = CPATCPACalculator(config)

    risk = risk_calc.assess_collision_risk(vessel_a, vessel_b)
    result = optimizer.optimize_avoidance(vessel_a, vessel_b, risk)

    print(f"Original CPA: {result.original_risk.cpa_distance:.3f} nmi")
    print(f"Best action: {result.best_action.description}")
    print(f"New CPA: {result.new_risk.cpa_distance:.3f} nmi")
    print(f"Improvement: {result.improvement:.3f} nmi")
    print(f"Success: {result.success}")
