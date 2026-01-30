"""
validators.py - Input Validation Utilities

This module provides comprehensive input validation for the seismic analysis application.
All user inputs should be validated before use to ensure numerical stability and correctness.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from exceptions import ValidationError


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    message: str = ""
    corrected_value: Any = None


# =============================================================================
# STRUCTURAL PARAMETER VALIDATORS
# =============================================================================

def validate_n_floors(n: int) -> ValidationResult:
    """Validate number of floors."""
    if not isinstance(n, (int, np.integer)):
        return ValidationResult(False, f"n_floors must be integer, got {type(n).__name__}")
    if n < 1:
        return ValidationResult(False, "n_floors must be at least 1", corrected_value=1)
    if n > 100:
        return ValidationResult(False, "n_floors exceeds maximum (100)", corrected_value=100)
    return ValidationResult(True)


def validate_floor_mass(mass: float) -> ValidationResult:
    """Validate floor mass in kg."""
    if not isinstance(mass, (int, float, np.number)):
        return ValidationResult(False, f"floor_mass must be numeric, got {type(mass).__name__}")
    if mass <= 0:
        return ValidationResult(False, "floor_mass must be positive", corrected_value=50000.0)
    if mass < 1000:  # Less than 1 tonne seems unrealistic
        return ValidationResult(False, "floor_mass seems too low (<1 tonne)")
    if mass > 1e7:  # More than 10,000 tonnes per floor
        return ValidationResult(False, "floor_mass exceeds realistic limit (10,000 tonnes)")
    return ValidationResult(True)


def validate_floor_height(height: float) -> ValidationResult:
    """Validate floor height in meters."""
    if not isinstance(height, (int, float, np.number)):
        return ValidationResult(False, f"floor_height must be numeric, got {type(height).__name__}")
    if height <= 0:
        return ValidationResult(False, "floor_height must be positive", corrected_value=3.0)
    if height < 2.0:
        return ValidationResult(False, "floor_height seems too low (<2m)")
    if height > 10.0:
        return ValidationResult(False, "floor_height exceeds typical limit (10m)")
    return ValidationResult(True)


def validate_stiffness(k: float) -> ValidationResult:
    """Validate column stiffness in N/m."""
    if not isinstance(k, (int, float, np.number)):
        return ValidationResult(False, f"stiffness must be numeric, got {type(k).__name__}")
    if k <= 0:
        return ValidationResult(False, "stiffness must be positive", corrected_value=1e7)
    if k < 1e4:  # Less than 10 kN/m
        return ValidationResult(False, "stiffness too low - structure unstable")
    if k > 1e12:  # More than 1 TN/m
        return ValidationResult(False, "stiffness exceeds realistic limit")
    return ValidationResult(True)


def validate_damping_ratio(zeta: float) -> ValidationResult:
    """Validate damping ratio (0 to 1)."""
    if not isinstance(zeta, (int, float, np.number)):
        return ValidationResult(False, f"damping_ratio must be numeric, got {type(zeta).__name__}")
    if zeta < 0:
        return ValidationResult(False, "damping_ratio cannot be negative", corrected_value=0.0)
    if zeta > 1.0:
        return ValidationResult(False, "damping_ratio cannot exceed 1.0 (100%)", corrected_value=1.0)
    if zeta > 0.5:
        return ValidationResult(True, "Warning: damping_ratio > 50% is unusual")
    return ValidationResult(True)


def validate_eccentricity(ecc: float) -> ValidationResult:
    """Validate eccentricity ratio (0 to 0.5)."""
    if not isinstance(ecc, (int, float, np.number)):
        return ValidationResult(False, f"eccentricity must be numeric, got {type(ecc).__name__}")
    if ecc < 0:
        return ValidationResult(False, "eccentricity cannot be negative", corrected_value=0.0)
    if ecc > 0.5:
        return ValidationResult(False, "eccentricity exceeds 50% - unrealistic", corrected_value=0.5)
    return ValidationResult(True)


# =============================================================================
# SEISMIC INPUT VALIDATORS
# =============================================================================

def validate_pga(pga: float) -> ValidationResult:
    """Validate peak ground acceleration in g."""
    if not isinstance(pga, (int, float, np.number)):
        return ValidationResult(False, f"pga must be numeric, got {type(pga).__name__}")
    if pga <= 0:
        return ValidationResult(False, "pga must be positive", corrected_value=0.1)
    if pga > 3.0:  # Highest recorded is ~2.7g
        return ValidationResult(False, "pga exceeds maximum recorded (3.0g)", corrected_value=3.0)
    return ValidationResult(True)


def validate_duration(duration: float) -> ValidationResult:
    """Validate earthquake duration in seconds."""
    if not isinstance(duration, (int, float, np.number)):
        return ValidationResult(False, f"duration must be numeric, got {type(duration).__name__}")
    if duration <= 0:
        return ValidationResult(False, "duration must be positive", corrected_value=10.0)
    if duration < 1.0:
        return ValidationResult(False, "duration too short (<1s)")
    if duration > 600.0:  # 10 minutes
        return ValidationResult(False, "duration exceeds limit (600s)", corrected_value=600.0)
    return ValidationResult(True)


def validate_time_step(dt: float, duration: float = None) -> ValidationResult:
    """Validate time step in seconds."""
    if not isinstance(dt, (int, float, np.number)):
        return ValidationResult(False, f"dt must be numeric, got {type(dt).__name__}")
    if dt <= 0:
        return ValidationResult(False, "dt must be positive", corrected_value=0.01)
    if dt > 0.1:
        return ValidationResult(False, "dt too large for accurate integration", corrected_value=0.05)
    if dt < 1e-4:
        return ValidationResult(False, "dt too small - excessive computation time")
    if duration and duration / dt > 1e6:
        return ValidationResult(False, "Too many time steps (>1M) - increase dt or reduce duration")
    return ValidationResult(True)


def validate_yield_drift(yield_drift: float) -> ValidationResult:
    """Validate yield drift ratio."""
    if not isinstance(yield_drift, (int, float, np.number)):
        return ValidationResult(False, f"yield_drift must be numeric")
    if yield_drift <= 0:
        return ValidationResult(False, "yield_drift must be positive", corrected_value=0.01)
    if yield_drift > 0.1:  # 10%
        return ValidationResult(False, "yield_drift too high (>10%)", corrected_value=0.05)
    if yield_drift < 0.001:  # 0.1%
        return ValidationResult(False, "yield_drift too low (<0.1%)")
    return ValidationResult(True)


def validate_hardening_ratio(alpha: float) -> ValidationResult:
    """Validate post-yield hardening ratio."""
    if not isinstance(alpha, (int, float, np.number)):
        return ValidationResult(False, f"hardening_ratio must be numeric")
    if alpha < 0:
        return ValidationResult(False, "hardening_ratio cannot be negative", corrected_value=0.0)
    if alpha > 0.5:
        return ValidationResult(False, "hardening_ratio too high (>0.5)", corrected_value=0.5)
    return ValidationResult(True)


# =============================================================================
# CONTROL DEVICE VALIDATORS
# =============================================================================

def validate_isolation_period(T_iso: float, T1: float = None) -> ValidationResult:
    """Validate isolation period in seconds."""
    if not isinstance(T_iso, (int, float, np.number)):
        return ValidationResult(False, f"isolation_period must be numeric")
    if T_iso <= 0:
        return ValidationResult(False, "isolation_period must be positive", corrected_value=2.5)
    if T_iso < 1.0:
        return ValidationResult(False, "isolation_period too short (<1s) - not effective")
    if T_iso > 6.0:
        return ValidationResult(False, "isolation_period too long (>6s)", corrected_value=5.0)
    if T1 and T_iso < 2.0 * T1:
        return ValidationResult(True, "Warning: isolation_period should be >2x structure period")
    return ValidationResult(True)


def validate_tmd_mass_ratio(ratio: float) -> ValidationResult:
    """Validate TMD mass ratio."""
    if not isinstance(ratio, (int, float, np.number)):
        return ValidationResult(False, f"tmd_mass_ratio must be numeric")
    if ratio <= 0:
        return ValidationResult(False, "tmd_mass_ratio must be positive", corrected_value=0.02)
    if ratio < 0.005:
        return ValidationResult(False, "tmd_mass_ratio too small (<0.5%) - not effective")
    if ratio > 0.1:
        return ValidationResult(False, "tmd_mass_ratio too large (>10%)", corrected_value=0.1)
    return ValidationResult(True)


# =============================================================================
# MATRIX VALIDATORS
# =============================================================================

def validate_matrix_dimensions(M: np.ndarray, K: np.ndarray, C: np.ndarray = None) -> ValidationResult:
    """Validate structural matrix dimensions are compatible."""
    if M.shape != K.shape:
        return ValidationResult(False, f"M and K dimension mismatch: {M.shape} vs {K.shape}")
    if M.shape[0] != M.shape[1]:
        return ValidationResult(False, f"M is not square: {M.shape}")
    if C is not None and C.shape != M.shape:
        return ValidationResult(False, f"C dimension mismatch: {C.shape}")
    return ValidationResult(True)


def validate_matrix_positive_definite(M: np.ndarray, name: str = "Matrix") -> ValidationResult:
    """Check if matrix is positive definite (all eigenvalues positive)."""
    try:
        eigvals = np.linalg.eigvalsh(M)
        if np.any(eigvals <= 0):
            return ValidationResult(
                False, 
                f"{name} is not positive definite. Min eigenvalue: {np.min(eigvals):.2e}"
            )
        if np.min(eigvals) / np.max(eigvals) < 1e-15:
            return ValidationResult(
                True, 
                f"Warning: {name} is nearly singular. Condition number: {np.max(eigvals)/np.min(eigvals):.2e}"
            )
        return ValidationResult(True)
    except Exception as e:
        return ValidationResult(False, f"Failed to analyze {name}: {str(e)}")


def validate_no_nan_inf(arr: np.ndarray, name: str = "Array") -> ValidationResult:
    """Check array for NaN or Inf values."""
    if np.any(np.isnan(arr)):
        nan_count = np.sum(np.isnan(arr))
        return ValidationResult(False, f"{name} contains {nan_count} NaN values")
    if np.any(np.isinf(arr)):
        inf_count = np.sum(np.isinf(arr))
        return ValidationResult(False, f"{name} contains {inf_count} Inf values")
    return ValidationResult(True)


# =============================================================================
# COMPREHENSIVE VALIDATION
# =============================================================================

def validate_structural_inputs(
    n_floors: int,
    floor_mass: float,
    floor_height: float,
    stiffness: float,
    damping_ratio: float,
    eccentricity: float = 0.0
) -> Tuple[bool, List[str]]:
    """
    Validate all structural input parameters.
    
    Returns:
        (is_valid, list of error/warning messages)
    """
    errors = []
    
    validators = [
        validate_n_floors(n_floors),
        validate_floor_mass(floor_mass),
        validate_floor_height(floor_height),
        validate_stiffness(stiffness),
        validate_damping_ratio(damping_ratio),
        validate_eccentricity(eccentricity),
    ]
    
    for result in validators:
        if not result.is_valid:
            errors.append(result.message)
        elif result.message:  # Warnings
            errors.append(result.message)
    
    is_valid = all(v.is_valid for v in validators)
    return is_valid, errors


def validate_seismic_inputs(
    pga: float,
    duration: float,
    dt: float
) -> Tuple[bool, List[str]]:
    """
    Validate all seismic input parameters.
    
    Returns:
        (is_valid, list of error/warning messages)
    """
    errors = []
    
    validators = [
        validate_pga(pga),
        validate_duration(duration),
        validate_time_step(dt, duration),
    ]
    
    for result in validators:
        if not result.is_valid:
            errors.append(result.message)
        elif result.message:
            errors.append(result.message)
    
    is_valid = all(v.is_valid for v in validators)
    return is_valid, errors


def validate_analysis_arrays(
    t: np.ndarray,
    ag: np.ndarray,
    M: np.ndarray,
    K: np.ndarray,
    C: np.ndarray
) -> Tuple[bool, List[str]]:
    """
    Validate arrays before running analysis.
    
    Returns:
        (is_valid, list of error/warning messages)
    """
    errors = []
    
    # Check dimensions
    dim_result = validate_matrix_dimensions(M, K, C)
    if not dim_result.is_valid:
        errors.append(dim_result.message)
        return False, errors
    
    # Check for NaN/Inf
    for arr, name in [(t, "Time array"), (ag, "Ground acceleration"), 
                      (M, "Mass matrix"), (K, "Stiffness matrix"), (C, "Damping matrix")]:
        result = validate_no_nan_inf(arr, name)
        if not result.is_valid:
            errors.append(result.message)
    
    # Check positive definiteness
    for mat, name in [(M, "Mass matrix"), (K, "Stiffness matrix")]:
        result = validate_matrix_positive_definite(mat, name)
        if not result.is_valid:
            errors.append(result.message)
        elif result.message:
            errors.append(result.message)
    
    # Check time array is monotonic
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        errors.append("Time array is not monotonically increasing")
    
    # Check array length consistency
    if len(t) != len(ag):
        errors.append(f"Time and ground acceleration length mismatch: {len(t)} vs {len(ag)}")
    
    is_valid = len([e for e in errors if not e.startswith("Warning")]) == 0
    return is_valid, errors
