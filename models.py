"""
models.py - Data Models and Enumerations for Seismic Analysis

This module contains all the type definitions, enumerations, and dataclasses
used throughout the seismic analysis application.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SeismicEvent(Enum):
    """
    Available ground motion types.
    
    IMPORTANT: All named events are SYNTHETICALLY GENERATED to match
    target spectral characteristics. They are NOT actual recorded motions.
    """
    KANAI_TAJIMI = "Synthetic - Kanai-Tajimi Spectrum"
    EL_CENTRO = "Synthetic - El Centro Spectrum Compatible"
    KOBE = "Synthetic - Kobe Spectrum Compatible"
    NORTHRIDGE = "Synthetic - Northridge Spectrum Compatible"
    CHILE = "Synthetic - Chile Subduction Compatible"
    MEXICO_CITY = "Synthetic - Mexico City Soft Soil"
    CHRISTCHURCH = "Synthetic - Christchurch Compatible"
    CUSTOM_SINE = "Educational - Sine Pulse"
    RICKER_WAVELET = "Educational - Ricker Wavelet"
    HARMONIC_SWEEP = "Educational - Frequency Sweep"
    MULTI_PULSE = "Educational - Multi-Frequency Pulse"


class DamperType(Enum):
    """Types of supplemental damping devices."""
    NONE = "No Additional Damping"
    TMD = "Tuned Mass Damper (TMD)"
    VISCOUS = "Viscous Dampers"
    FRICTION = "Friction Dampers"
    VISCOELASTIC = "Viscoelastic Dampers"
    MR_DAMPER = "Magnetorheological (MR) Damper"
    INERTER = "Tuned Inerter Damper"


class AnalysisType(Enum):
    """Types of structural analysis."""
    LINEAR = "Linear Elastic"
    NONLINEAR_BILINEAR = "Nonlinear Bilinear"
    NONLINEAR_TAKEDA = "Nonlinear Takeda Hysteresis"
    PUSHOVER = "Static Pushover"


class PerformanceLevel(Enum):
    """ASCE 41 Performance Levels with drift limits."""
    OPERATIONAL = ("Operational", 0.5, "#22c55e")
    IMMEDIATE_OCCUPANCY = ("Immediate Occupancy", 1.0, "#84cc16")
    LIFE_SAFETY = ("Life Safety", 2.0, "#eab308")
    COLLAPSE_PREVENTION = ("Collapse Prevention", 4.0, "#f97316")
    COLLAPSE = ("Collapse", float('inf'), "#ef4444")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StructuralProperties:
    """
    Properties defining the structural system.
    
    Attributes
    ----------
    n_floors : int
        Number of floors (1-100)
    floor_mass : float
        Mass per floor in kg (must be positive)
    floor_height : float
        Story height in meters (must be positive)
    building_width : float
        Building width in meters
    building_depth : float
        Building depth in meters
    column_stiffness : float
        Column stiffness in N/m
    damping_ratio : float
        Inherent damping ratio (0-1)
    eccentricity : float
        Mass eccentricity ratio (0-0.5)
    yield_drift : float
        Yield drift ratio for nonlinear analysis
    hardening_ratio : float
        Post-yield stiffness ratio (0-1)
    """
    n_floors: int = 8
    floor_mass: float = 50000.0  # kg
    floor_height: float = 3.0  # m
    building_width: float = 12.0  # m
    building_depth: float = 12.0  # m
    column_stiffness: float = 1e7  # N/m
    damping_ratio: float = 0.05
    eccentricity: float = 0.0  # ratio
    yield_drift: float = 0.01  # for nonlinear
    hardening_ratio: float = 0.02  # post-yield stiffness ratio
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.n_floors < 1:
            raise ValueError(f"n_floors must be >= 1, got {self.n_floors}")
        if self.floor_mass <= 0:
            raise ValueError(f"floor_mass must be > 0, got {self.floor_mass}")
        if self.floor_height <= 0:
            raise ValueError(f"floor_height must be > 0, got {self.floor_height}")
        if not 0 <= self.damping_ratio <= 1:
            raise ValueError(f"damping_ratio must be 0-1, got {self.damping_ratio}")
        if not 0 <= self.eccentricity <= 0.5:
            raise ValueError(f"eccentricity must be 0-0.5, got {self.eccentricity}")


@dataclass
class SeismicInput:
    """Parameters defining the seismic excitation."""
    event_type: SeismicEvent = SeismicEvent.KANAI_TAJIMI
    pga: float = 0.4  # g
    duration: float = 20.0  # s
    soil_type: str = "medium"
    direction: str = "X"  # X, Y, XY
    vertical_component: bool = False
    scale_factor: float = 1.0


@dataclass
class ControlDevices:
    """Configuration for structural control devices."""
    base_isolated: bool = False
    isolation_period: float = 2.5  # s
    isolation_damping: float = 0.15
    damper_type: DamperType = DamperType.NONE
    tmd_mass_ratio: float = 0.02
    tmd_floor: int = -1  # top floor
    viscous_alpha: float = 0.5
    viscous_c: float = 1e5


@dataclass
class SimulationResults:
    """Container for all simulation results."""
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    ground_acc: np.ndarray = field(default_factory=lambda: np.array([]))
    displacements: np.ndarray = field(default_factory=lambda: np.array([]))
    velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    accelerations: np.ndarray = field(default_factory=lambda: np.array([]))
    base_shear: np.ndarray = field(default_factory=lambda: np.array([]))
    periods: np.ndarray = field(default_factory=lambda: np.array([]))
    mode_shapes: np.ndarray = field(default_factory=lambda: np.array([]))
    energy: Dict[str, np.ndarray] = field(default_factory=dict)
    damage_index: np.ndarray = field(default_factory=lambda: np.array([]))
    fragility_data: Dict = field(default_factory=dict)
    hysteresis: Dict = field(default_factory=dict)
    response_spectra: Dict = field(default_factory=dict)
    comparison_data: Dict = field(default_factory=dict)
    statistics: Dict = field(default_factory=dict)
    residual_drift: np.ndarray = field(default_factory=lambda: np.array([]))


# =============================================================================
# EARTHQUAKE DATABASE
# =============================================================================

EARTHQUAKE_DATABASE: Dict[SeismicEvent, Dict[str, Any]] = {
    SeismicEvent.EL_CENTRO: {
        "pga": 0.35, 
        "duration": 40, 
        "freq_content": "broadband", 
        "magnitude": 6.9,
        "description": "1940 Imperial Valley - Broadband content"
    },
    SeismicEvent.KOBE: {
        "pga": 0.82, 
        "duration": 20, 
        "freq_content": "near_field", 
        "magnitude": 6.9,
        "description": "1995 Hyogoken-Nanbu - Near-field pulse"
    },
    SeismicEvent.NORTHRIDGE: {
        "pga": 0.84, 
        "duration": 20, 
        "freq_content": "near_field", 
        "magnitude": 6.7,
        "description": "1994 Northridge - Near-field directivity"
    },
    SeismicEvent.CHILE: {
        "pga": 0.65, 
        "duration": 180, 
        "freq_content": "long_duration", 
        "magnitude": 8.8,
        "description": "2010 Maule - Subduction megathrust"
    },
    SeismicEvent.MEXICO_CITY: {
        "pga": 0.17, 
        "duration": 180, 
        "freq_content": "soft_soil", 
        "magnitude": 8.0,
        "description": "1985 Michoacan - Soft soil amplification"
    },
    SeismicEvent.CHRISTCHURCH: {
        "pga": 0.80, 
        "duration": 25, 
        "freq_content": "near_field", 
        "magnitude": 6.3,
        "description": "2011 Canterbury - Shallow crustal"
    },
}


# =============================================================================
# IS 1893 ZONE FACTORS
# =============================================================================

IS_ZONE_FACTORS: Dict[str, float] = {
    "II": 0.10,
    "III": 0.16,
    "IV": 0.24,
    "V": 0.36,
}
