"""
CPA (Closest Point of Approach) and TCPA (Time to CPA) calculations for collision risk assessment.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class VesselState:
    """Vessel state information."""
    mmsi: int
    timestamp: float
    lat: float
    lon: float
    x: float  # Local x coordinate (meters)
    y: float  # Local y coordinate (meters)
    sog: float  # Speed over ground (knots)
    cog: float  # Course over ground (degrees)
    heading: Optional[float] = None
    length: Optional[float] = None
    width: Optional[float] = None
    ship_type: Optional[int] = None


@dataclass
class EncounterGeometry:
    """Encounter geometry parameters."""
    cpa_distance: float  # CPA distance (nautical miles)
    tcpa: float  # Time to CPA (minutes)
    dcpa: float  # Distance at CPA (nautical miles)
    bearing_target_to_other: float  # Bearing from target to other vessel (degrees)
    bearing_other_to_target: float  # Bearing from other to target vessel (degrees)
    relative_speed: float  # Relative speed (knots)
    encounter_type: str  # Type of encounter (head-on, crossing, overtaking)


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    target_mmsi: int
    other_mmsi: int
    risk_level: RiskLevel
    cpa_distance: float
    tcpa: float
    encounter_geometry: EncounterGeometry
    colregs_situation: str
    recommended_action: str
    confidence: float


class CPATCPACalculator:
    """Calculator for CPA and TCPA between vessels."""

    def __init__(self, config: Dict):
        """
        Initialize CPA/TCPA calculator.

        Args:
            config: Configuration dictionary with risk assessment parameters
        """
        self.config = config
        risk_config = config.get('simulation', {}).get('risk', {})

        # Risk thresholds
        self.safe_distance = risk_config.get('safe_distance', 0.5)  # nautical miles
        self.time_horizon = risk_config.get('time_horizon', 1800)   # seconds (30 minutes)
        self.update_interval = risk_config.get('update_interval', 30)  # seconds

        # Conversion factors
        self.knots_to_ms = 0.514444  # knots to m/s
        self.nmi_to_m = 1852.0       # nautical miles to meters
        self.deg_to_rad = np.pi / 180

    def calculate_cpa_tcpa(self, vessel_a: VesselState, vessel_b: VesselState) -> Tuple[float, float]:
        """
        Calculate CPA distance and TCPA between two vessels.

        Args:
            vessel_a: State of first vessel
            vessel_b: State of second vessel

        Returns:
            Tuple of (CPA distance in nautical miles, TCPA in minutes)
        """
        # Position vectors (in meters)
        pos_a = np.array([vessel_a.x, vessel_a.y])
        pos_b = np.array([vessel_b.x, vessel_b.y])

        # Velocity vectors (in m/s)
        vel_a = self._sog_cog_to_velocity(vessel_a.sog, vessel_a.cog)
        vel_b = self._sog_cog_to_velocity(vessel_b.sog, vessel_b.cog)

        # Relative position and velocity
        delta_pos = pos_a - pos_b
        delta_vel = vel_a - vel_b

        # Calculate TCPA
        delta_vel_squared = np.dot(delta_vel, delta_vel)

        if delta_vel_squared < 1e-6:  # Vessels have same velocity
            tcpa_seconds = 0.0
            cpa_distance_m = np.linalg.norm(delta_pos)
        else:
            tcpa_seconds = max(0.0, -np.dot(delta_pos, delta_vel) / delta_vel_squared)

            # Calculate CPA distance
            cpa_position = delta_pos + delta_vel * tcpa_seconds
            cpa_distance_m = np.linalg.norm(cpa_position)

        # Convert units
        cpa_distance_nmi = cpa_distance_m / self.nmi_to_m
        tcpa_minutes = tcpa_seconds / 60.0

        return cpa_distance_nmi, tcpa_minutes

    def _sog_cog_to_velocity(self, sog: float, cog: float) -> np.array:
        """
        Convert SOG and COG to velocity vector.

        Args:
            sog: Speed over ground (knots)
            cog: Course over ground (degrees)

        Returns:
            Velocity vector [vx, vy] in m/s
        """
        speed_ms = sog * self.knots_to_ms
        cog_rad = cog * self.deg_to_rad

        # COG is measured clockwise from North
        # Convert to standard math convention (counter-clockwise from East)
        vx = speed_ms * np.sin(cog_rad)  # East component
        vy = speed_ms * np.cos(cog_rad)  # North component

        return np.array([vx, vy])

    def calculate_encounter_geometry(self, vessel_a: VesselState,
                                   vessel_b: VesselState) -> EncounterGeometry:
        """
        Calculate detailed encounter geometry.

        Args:
            vessel_a: Target vessel state
            vessel_b: Other vessel state

        Returns:
            EncounterGeometry object with detailed parameters
        """
        # Basic CPA/TCPA
        cpa_distance, tcpa = self.calculate_cpa_tcpa(vessel_a, vessel_b)

        # Relative bearings
        bearing_a_to_b = self._calculate_bearing(vessel_a, vessel_b)
        bearing_b_to_a = self._calculate_bearing(vessel_b, vessel_a)

        # Relative speed
        vel_a = self._sog_cog_to_velocity(vessel_a.sog, vessel_a.cog)
        vel_b = self._sog_cog_to_velocity(vessel_b.sog, vessel_b.cog)
        relative_velocity = vel_a - vel_b
        relative_speed_ms = np.linalg.norm(relative_velocity)
        relative_speed_knots = relative_speed_ms / self.knots_to_ms

        # Encounter type classification
        encounter_type = self._classify_encounter_type(vessel_a, vessel_b, bearing_a_to_b)

        return EncounterGeometry(
            cpa_distance=cpa_distance,
            tcpa=tcpa,
            dcpa=cpa_distance,  # Same as CPA for this implementation
            bearing_target_to_other=bearing_a_to_b,
            bearing_other_to_target=bearing_b_to_a,
            relative_speed=relative_speed_knots,
            encounter_type=encounter_type
        )

    def _calculate_bearing(self, from_vessel: VesselState, to_vessel: VesselState) -> float:
        """Calculate bearing from one vessel to another."""
        dx = to_vessel.x - from_vessel.x
        dy = to_vessel.y - from_vessel.y

        bearing_rad = np.arctan2(dx, dy)  # North = 0, East = π/2
        bearing_deg = np.rad2deg(bearing_rad)
        bearing_deg = (bearing_deg + 360) % 360  # Normalize to [0, 360)

        return bearing_deg

    def _classify_encounter_type(self, vessel_a: VesselState, vessel_b: VesselState,
                               bearing_a_to_b: float) -> str:
        """
        Classify encounter type based on COLREGs.

        Args:
            vessel_a: Target vessel
            vessel_b: Other vessel
            bearing_a_to_b: Bearing from A to B

        Returns:
            Encounter type: 'head-on', 'crossing', 'overtaking'
        """
        # Relative bearing of B from A's perspective
        relative_bearing = (bearing_a_to_b - vessel_a.cog + 360) % 360

        # Course difference
        course_diff = abs(vessel_a.cog - vessel_b.cog)
        course_diff = min(course_diff, 360 - course_diff)

        # Classification logic
        if course_diff > 165:  # Nearly opposite courses
            return 'head-on'
        elif relative_bearing > 135 and relative_bearing < 225:  # Behind target
            if vessel_b.sog > vessel_a.sog:  # B is faster
                return 'overtaking'
            else:
                return 'crossing'
        else:
            return 'crossing'

    def assess_collision_risk(self, vessel_a: VesselState,
                            vessel_b: VesselState) -> RiskAssessment:
        """
        Assess collision risk between two vessels.

        Args:
            vessel_a: Target vessel state
            vessel_b: Other vessel state

        Returns:
            RiskAssessment object
        """
        # Calculate encounter geometry
        geometry = self.calculate_encounter_geometry(vessel_a, vessel_b)

        # Determine risk level
        risk_level = self._determine_risk_level(geometry)

        # COLREGs situation analysis
        colregs_situation = self._analyze_colregs_situation(vessel_a, vessel_b, geometry)

        # Recommended action
        recommended_action = self._recommend_action(vessel_a, vessel_b, geometry, colregs_situation)

        # Confidence assessment
        confidence = self._assess_confidence(vessel_a, vessel_b, geometry)

        return RiskAssessment(
            target_mmsi=vessel_a.mmsi,
            other_mmsi=vessel_b.mmsi,
            risk_level=risk_level,
            cpa_distance=geometry.cpa_distance,
            tcpa=geometry.tcpa,
            encounter_geometry=geometry,
            colregs_situation=colregs_situation,
            recommended_action=recommended_action,
            confidence=confidence
        )

    def _determine_risk_level(self, geometry: EncounterGeometry) -> RiskLevel:
        """Determine risk level based on CPA and TCPA."""
        cpa = geometry.cpa_distance
        tcpa = geometry.tcpa

        # No risk if TCPA is negative (vessels diverging) or too far in future
        if tcpa <= 0 or tcpa > self.time_horizon / 60:
            return RiskLevel.SAFE

        # Risk assessment based on CPA distance and TCPA
        if cpa >= 2.0:  # > 2 nautical miles
            return RiskLevel.SAFE
        elif cpa >= 1.0:  # 1-2 nautical miles
            if tcpa < 5:  # < 5 minutes
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        elif cpa >= 0.5:  # 0.5-1 nautical miles
            if tcpa < 3:  # < 3 minutes
                return RiskLevel.HIGH
            elif tcpa < 10:  # 3-10 minutes
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        else:  # < 0.5 nautical miles
            if tcpa < 2:  # < 2 minutes
                return RiskLevel.CRITICAL
            elif tcpa < 5:  # 2-5 minutes
                return RiskLevel.HIGH
            else:
                return RiskLevel.MEDIUM

    def _analyze_colregs_situation(self, vessel_a: VesselState, vessel_b: VesselState,
                                 geometry: EncounterGeometry) -> str:
        """Analyze COLREGs situation to determine give-way/stand-on roles."""
        encounter_type = geometry.encounter_type
        bearing_a_to_b = geometry.bearing_target_to_other

        if encounter_type == 'head-on':
            return 'head-on: both vessels give-way (turn starboard)'

        elif encounter_type == 'overtaking':
            if vessel_b.sog > vessel_a.sog:
                return 'overtaking: other vessel (B) gives way'
            else:
                return 'overtaking: target vessel (A) gives way'

        elif encounter_type == 'crossing':
            # Determine relative bearing
            relative_bearing = (bearing_a_to_b - vessel_a.cog + 360) % 360

            if relative_bearing < 180:  # Other vessel on starboard side
                return 'crossing: target vessel (A) gives way'
            else:  # Other vessel on port side
                return 'crossing: other vessel (B) gives way'

        return 'unknown situation'

    def _recommend_action(self, vessel_a: VesselState, vessel_b: VesselState,
                         geometry: EncounterGeometry, colregs_situation: str) -> str:
        """Recommend action based on risk assessment."""
        risk_level = self._determine_risk_level(geometry)

        if risk_level == RiskLevel.SAFE:
            return 'maintain course and speed'

        elif risk_level == RiskLevel.LOW:
            return 'monitor situation closely'

        elif risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]:
            if 'target vessel (A) gives way' in colregs_situation:
                if geometry.encounter_type == 'crossing':
                    return 'alter course to starboard or reduce speed'
                else:
                    return 'alter course or speed to increase CPA'
            else:
                return 'maintain course and speed (stand-on vessel)'

        elif risk_level == RiskLevel.CRITICAL:
            return 'immediate evasive action required'

        return 'assess situation manually'

    def _assess_confidence(self, vessel_a: VesselState, vessel_b: VesselState,
                          geometry: EncounterGeometry) -> float:
        """Assess confidence in risk assessment."""
        confidence = 1.0

        # Reduce confidence for very low speeds (data quality issues)
        if vessel_a.sog < 1.0 or vessel_b.sog < 1.0:
            confidence *= 0.7

        # Reduce confidence for very long TCPA (prediction uncertainty)
        if geometry.tcpa > 20:  # > 20 minutes
            confidence *= 0.8

        # Reduce confidence for very close encounters (measurement noise impact)
        if geometry.cpa_distance < 0.1:  # < 0.1 nautical miles
            confidence *= 0.9

        return max(0.0, min(1.0, confidence))


class MultiVesselRiskAssessor:
    """Multi-vessel risk assessment manager."""

    def __init__(self, config: Dict):
        """Initialize multi-vessel risk assessor."""
        self.calculator = CPATCPACalculator(config)
        self.config = config

    def assess_all_pairs(self, vessels: List[VesselState]) -> List[RiskAssessment]:
        """
        Assess collision risk for all vessel pairs.

        Args:
            vessels: List of vessel states

        Returns:
            List of risk assessments for all pairs
        """
        assessments = []

        for i in range(len(vessels)):
            for j in range(i + 1, len(vessels)):
                try:
                    assessment = self.calculator.assess_collision_risk(vessels[i], vessels[j])
                    assessments.append(assessment)
                except Exception as e:
                    logger.warning(f"Failed to assess risk between {vessels[i].mmsi} and {vessels[j].mmsi}: {e}")

        return assessments

    def get_high_risk_encounters(self, assessments: List[RiskAssessment],
                               min_risk_level: RiskLevel = RiskLevel.MEDIUM) -> List[RiskAssessment]:
        """Filter assessments to high-risk encounters only."""
        risk_order = {
            RiskLevel.SAFE: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }

        min_level = risk_order[min_risk_level]
        high_risk = [a for a in assessments if risk_order[a.risk_level] >= min_level]

        # Sort by risk level (highest first) then by TCPA (shortest first)
        high_risk.sort(key=lambda x: (-risk_order[x.risk_level], x.tcpa))

        return high_risk

    def create_risk_summary(self, assessments: List[RiskAssessment]) -> Dict:
        """Create summary statistics of risk assessments."""
        if not assessments:
            return {}

        risk_counts = {}
        for level in RiskLevel:
            risk_counts[level.value] = sum(1 for a in assessments if a.risk_level == level)

        cpa_values = [a.cpa_distance for a in assessments]
        tcpa_values = [a.tcpa for a in assessments if a.tcpa > 0]

        summary = {
            'total_encounters': len(assessments),
            'risk_level_counts': risk_counts,
            'cpa_statistics': {
                'mean': np.mean(cpa_values) if cpa_values else 0,
                'min': np.min(cpa_values) if cpa_values else 0,
                'max': np.max(cpa_values) if cpa_values else 0,
                'std': np.std(cpa_values) if cpa_values else 0
            },
            'tcpa_statistics': {
                'mean': np.mean(tcpa_values) if tcpa_values else 0,
                'min': np.min(tcpa_values) if tcpa_values else 0,
                'max': np.max(tcpa_values) if tcpa_values else 0,
                'std': np.std(tcpa_values) if tcpa_values else 0
            }
        }

        return summary


if __name__ == "__main__":
    # Example usage
    config = {
        'simulation': {
            'risk': {
                'safe_distance': 0.5,
                'time_horizon': 1800,
                'update_interval': 30
            }
        }
    }

    # Create sample vessels
    vessel_a = VesselState(
        mmsi=123456789,
        timestamp=0,
        lat=40.7128,
        lon=-74.0060,
        x=0,
        y=0,
        sog=15.0,
        cog=90.0
    )

    vessel_b = VesselState(
        mmsi=987654321,
        timestamp=0,
        lat=40.7138,
        lon=-74.0050,
        x=1000,
        y=1000,
        sog=12.0,
        cog=270.0
    )

    # Test CPA/TCPA calculation
    calculator = CPATCPACalculator(config)
    cpa, tcpa = calculator.calculate_cpa_tcpa(vessel_a, vessel_b)
    print(f"CPA: {cpa:.3f} nmi, TCPA: {tcpa:.2f} min")

    # Test risk assessment
    assessment = calculator.assess_collision_risk(vessel_a, vessel_b)
    print(f"Risk level: {assessment.risk_level.value}")
    print(f"Recommended action: {assessment.recommended_action}")
    print(f"COLREGs situation: {assessment.colregs_situation}")
