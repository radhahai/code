"""
design_codes.py - Design Code Calculations

This module contains design code implementations:
- IS 1893:2016 (India)
- Other codes can be added here
"""

import numpy as np
from typing import Tuple, Dict

from models import IS_ZONE_FACTORS


# =============================================================================
# IS 1893:2016 SPECTRUM
# =============================================================================

def compute_is1893_spectrum(
    periods: np.ndarray,
    soil: str
) -> np.ndarray:
    """
    IS 1893:2016 design spectrum for 5% damping.
    
    Parameters
    ----------
    periods : np.ndarray
        Period array in seconds
    soil : str
        Soil type: "Rock", "Medium", "Soft"
    
    Returns
    -------
    np.ndarray
        Sa/g array (spectral acceleration / g)
    """
    soil = soil.lower()
    Sa = np.zeros_like(periods, dtype=float)
    
    for i, T in enumerate(periods):
        if T <= 0:
            Sa[i] = 0.0
            continue
        
        if soil == "soft":
            # Type III soil
            if T <= 0.1:
                Sa[i] = 1 + 15 * T
            elif T <= 0.67:
                Sa[i] = 2.5
            else:
                Sa[i] = 1.67 / T
        elif soil == "rock":
            # Type I soil (Rock/Hard soil)
            if T <= 0.1:
                Sa[i] = 1 + 15 * T
            elif T <= 0.4:
                Sa[i] = 2.5
            else:
                Sa[i] = 1.0 / T
        else:
            # Type II soil (Medium)
            if T <= 0.1:
                Sa[i] = 1 + 15 * T
            elif T <= 0.55:
                Sa[i] = 2.5
            else:
                Sa[i] = 1.36 / T
    
    return Sa


def compute_design_base_shear(
    total_weight: float,
    T1: float,
    soil: str,
    zone: str,
    importance: float,
    response_reduction: float
) -> Tuple[float, float]:
    """
    Compute design base shear per IS 1893:2016.
    
    Vb = Ah × W
    
    where:
    Ah = (Z/2) × (I/R) × (Sa/g)
    
    Parameters
    ----------
    total_weight : float
        Total seismic weight in N
    T1 : float
        Fundamental period in seconds
    soil : str
        Soil type
    zone : str
        Seismic zone (II, III, IV, V)
    importance : float
        Importance factor (I)
    response_reduction : float
        Response reduction factor (R)
    
    Returns
    -------
    tuple
        (Ah, Vb) - Horizontal seismic coefficient and design base shear
    """
    Z = IS_ZONE_FACTORS.get(zone, 0.16)
    Sa = compute_is1893_spectrum(np.array([max(T1, 0.01)]), soil)[0]
    
    # Horizontal seismic coefficient
    Ah = (Z / 2) * (importance / max(response_reduction, 1e-3)) * Sa
    
    # Design base shear
    Vb = Ah * total_weight
    
    return Ah, Vb


def get_drift_limit_is1893() -> float:
    """
    Get allowable drift limit per IS 1893:2016.
    
    For buildings with brittle non-structural elements:
    δ/h ≤ 0.004 (0.4%)
    
    Returns
    -------
    float
        Drift limit in percentage
    """
    return 0.4


def get_zone_factor(zone: str) -> float:
    """
    Get zone factor Z for given seismic zone.
    
    Parameters
    ----------
    zone : str
        Seismic zone (II, III, IV, V)
    
    Returns
    -------
    float
        Zone factor Z
    """
    return IS_ZONE_FACTORS.get(zone, 0.16)


# =============================================================================
# DAMPING CORRECTION
# =============================================================================

def damping_correction_factor(damping: float, code: str = "IS1893") -> float:
    """
    Compute damping correction factor for spectra.
    
    IS 1893 uses 5% damping. For other damping values,
    multiply Sa by correction factor.
    
    Parameters
    ----------
    damping : float
        Damping ratio
    code : str
        Design code
    
    Returns
    -------
    float
        Correction factor
    """
    # Eurocode 8 formula (commonly used)
    eta = np.sqrt(10 / (5 + damping * 100))
    return max(eta, 0.55)


# =============================================================================
# RESPONSE MODIFICATION
# =============================================================================

def get_response_reduction_factor(
    system_type: str,
    code: str = "IS1893"
) -> Dict[str, float]:
    """
    Get response reduction factors for different structural systems.
    
    Parameters
    ----------
    system_type : str
        Structural system type
    code : str
        Design code
    
    Returns
    -------
    dict
        R factor and other parameters
    """
    # IS 1893:2016 Table 9
    systems = {
        "OMRF": {"R": 3.0, "desc": "Ordinary Moment Resisting Frame"},
        "SMRF": {"R": 5.0, "desc": "Special Moment Resisting Frame"},
        "OCBF": {"R": 4.0, "desc": "Ordinary Concentrically Braced Frame"},
        "SCBF": {"R": 4.5, "desc": "Special Concentrically Braced Frame"},
        "EBF": {"R": 5.0, "desc": "Eccentrically Braced Frame"},
        "SW": {"R": 4.0, "desc": "Shear Wall"},
        "DUAL": {"R": 5.0, "desc": "Dual System (Frame + Wall)"},
    }
    
    return systems.get(system_type, {"R": 3.0, "desc": "Default"})


# =============================================================================
# VERTICAL DISTRIBUTION
# =============================================================================

def compute_vertical_distribution(
    Vb: float,
    floor_masses: np.ndarray,
    floor_heights: np.ndarray,
    T1: float
) -> np.ndarray:
    """
    Compute vertical distribution of base shear.
    
    IS 1893 uses: Qi = Vb × (Wi × hi²) / Σ(Wj × hj²)
    
    Parameters
    ----------
    Vb : float
        Design base shear
    floor_masses : np.ndarray
        Mass at each floor
    floor_heights : np.ndarray
        Height of each floor from base
    T1 : float
        Fundamental period
    
    Returns
    -------
    np.ndarray
        Lateral force at each floor
    """
    # IS 1893 distribution (parabolic)
    Wi_hi2 = floor_masses * floor_heights**2
    sum_Wi_hi2 = np.sum(Wi_hi2)
    
    if sum_Wi_hi2 > 0:
        Q = Vb * Wi_hi2 / sum_Wi_hi2
    else:
        Q = np.zeros_like(floor_masses)
    
    return Q
