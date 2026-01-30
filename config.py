"""
config.py - Configuration Management

This module handles all configuration, defaults, and session state management
for the seismic analysis application.
"""

from typing import Dict, Any
import streamlit as st

from models import SeismicEvent, DamperType, AnalysisType


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    # Structure
    "n_floors": 10,
    "floor_height": 3.0,
    "floor_mass_tonnes": 50,
    "building_width": 15.0,
    "building_depth": 15.0,
    "stiffness_factor": 1.0,
    "eccentricity_pct": 5,
    "irregularity": "Regular",
    
    # Seismic Input
    "event_selected": SeismicEvent.KANAI_TAJIMI.value,
    "pga": 0.4,
    "duration": 25,
    "soil": "medium",
    "direction": "X",
    "scale_factor": 1.0,
    
    # Damping & Control
    "damping_pct": 5.0,
    "base_iso": False,
    "iso_period": 2.5,
    "iso_damping": 0.15,
    "damper_selected": DamperType.NONE.value,
    "tmd_mass_ratio": 0.02,
    "tmd_floor": 10,
    "viscous_c_kns": 1000,
    
    # Analysis
    "analysis_selected": AnalysisType.LINEAR.value,
    "yield_drift_pct": 1.0,
    "hardening": 0.02,
    "dt": 0.02,
    
    # Output Options
    "compute_spectra": True,
    "compute_fragility": False,
    "compare_mode": False,
    
    # Reproducibility
    "seed_enabled": False,
    "random_seed": 42,
    
    # IS 1893 Code
    "is_code_enabled": True,
    "is_zone": "III",
    "is_importance": 1.0,
    "is_response_reduction": 5.0,
    "is_soil": "Medium",
    "show_is_overlay": True,
}


# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def ensure_config_state() -> None:
    """Initialize all session state keys with defaults if not present."""
    for key, value in DEFAULT_CONFIG.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_config(config: Dict[str, Any]) -> None:
    """Apply a configuration dictionary to session state."""
    for key in DEFAULT_CONFIG:
        if key in config:
            st.session_state[key] = config[key]


def build_config_snapshot() -> Dict[str, Any]:
    """Build a snapshot of current configuration."""
    return {key: st.session_state.get(key, DEFAULT_CONFIG[key]) for key in DEFAULT_CONFIG}


def reset_config() -> None:
    """Reset configuration to defaults."""
    apply_config(DEFAULT_CONFIG)


# =============================================================================
# STIFFNESS SCALING
# =============================================================================

def compute_scaled_stiffness(
    n_floors: int,
    floor_mass: float,
    stiffness_factor: float = 1.0,
    target_period_coefficient: float = 0.1
) -> float:
    """
    Compute column stiffness scaled based on number of floors.
    
    Rule: For realistic behavior, taller buildings need stiffer structural systems.
    Target: A 10-story building should have T1 ≈ 1.0 - 1.5s
    
    Using approximate formula: T1 ≈ 0.1 * N (for moment frames)
    This gives us the target stiffness to achieve realistic periods.
    
    Parameters
    ----------
    n_floors : int
        Number of floors
    floor_mass : float
        Mass per floor in kg
    stiffness_factor : float
        User-adjustable multiplier
    target_period_coefficient : float
        Coefficient for T = coeff * N formula (default 0.1)
    
    Returns
    -------
    float
        Column stiffness in N/m
    """
    import numpy as np
    
    # Target fundamental period based on building height
    # T1 ≈ 0.1 * N for moment frames (ASCE 7 approximate)
    T_target = target_period_coefficient * n_floors
    
    # Clamp to realistic range
    T_target = max(0.3, min(T_target, 4.0))
    
    # For a shear building, T1 ≈ 2π / ω1
    # ω1 ≈ sqrt(k_eff / m_eff) where m_eff ≈ 0.7 * total_mass for first mode
    # Solving for k: k ≈ m_eff * (2π/T)²
    
    total_mass = n_floors * floor_mass
    effective_mass = 0.7 * total_mass  # First mode participation
    omega_target = 2 * np.pi / T_target
    
    # Stiffness per floor (approximate for shear building)
    k_total = effective_mass * omega_target**2
    k_per_floor = k_total / n_floors * 2.5  # Factor for shear building behavior
    
    # Apply user factor
    k_scaled = k_per_floor * stiffness_factor
    
    # Ensure minimum stiffness
    k_min = 1e6  # 1 MN/m minimum
    k_max = 1e9  # 1 GN/m maximum
    
    return max(k_min, min(k_scaled, k_max))


def check_period_warning(T1: float, n_floors: int) -> str:
    """
    Check if fundamental period is realistic and return warning if not.
    
    Parameters
    ----------
    T1 : float
        Fundamental period in seconds
    n_floors : int
        Number of floors
    
    Returns
    -------
    str
        Warning message or empty string if OK
    """
    # Expected range based on building height
    T_expected_low = 0.05 * n_floors
    T_expected_high = 0.15 * n_floors
    
    if T1 > 5.0:
        return f"⚠️ WARNING: Fundamental period T₁ = {T1:.2f}s is unrealistically long (>5s). The building is too flexible. Consider increasing stiffness."
    elif T1 > T_expected_high * 1.5:
        return f"⚠️ CAUTION: Period T₁ = {T1:.2f}s is longer than expected for a {n_floors}-story building. Expected range: {T_expected_low:.1f}s - {T_expected_high:.1f}s"
    elif T1 < T_expected_low * 0.5:
        return f"ℹ️ NOTE: Period T₁ = {T1:.2f}s is shorter than typical. The building is very stiff."
    
    return ""
