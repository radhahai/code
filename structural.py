"""
structural.py - Structural Matrix Assembly

This module handles the construction of mass, stiffness, and damping
matrices for the MDOF shear building model.

Includes:
- Mass, stiffness, damping matrix generation
- Modal analysis
- Support for TMD, base isolation, and viscous dampers
- Comprehensive validation and error handling
"""

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import eigh
import warnings

from models import StructuralProperties, ControlDevices, DamperType
from exceptions import MatrixError, ValidationError
from logging_config import get_logger

logger = get_logger("structural")

# Numerical constants
MIN_STIFFNESS = 1e3  # Minimum stiffness value (N/m)
MIN_MASS = 1.0  # Minimum mass value (kg)


# =============================================================================
# MATRIX ASSEMBLY
# =============================================================================

def build_structural_matrices(
    props: StructuralProperties,
    controls: ControlDevices
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct Mass (M), Stiffness (K), and preliminary Damping (C) matrices
    for 3D Shear Building with advanced features.
    
    DOFs per floor: Ux, Uy, Rz (translation X, Y, rotation Z)
    
    Parameters
    ----------
    props : StructuralProperties
        Structural properties of the building
    controls : ControlDevices
        Control device configuration
    
    Returns
    -------
    M : np.ndarray
        Mass matrix
    K : np.ndarray
        Stiffness matrix
    C : np.ndarray
        Preliminary damping matrix (zeros, will be computed with Rayleigh)
    
    Raises
    ------
    ValidationError
        If input parameters are invalid
    MatrixError
        If matrix assembly fails
    """
    # Validate inputs
    n = props.n_floors
    if n < 1:
        raise ValidationError("n_floors", n, ">= 1")
    
    m = props.floor_mass
    if m <= 0:
        raise ValidationError("floor_mass", m, "> 0")
    
    h = props.floor_height
    if h <= 0:
        raise ValidationError("floor_height", h, "> 0")
    
    w = props.building_width
    d = props.building_depth
    k_col = max(props.column_stiffness, MIN_STIFFNESS)  # Ensure minimum stiffness
    ecc = props.eccentricity * w  # Convert ratio to actual distance
    
    logger.debug(f"Building matrices: {n} floors, mass={m:.0f}kg, k={k_col:.2e}N/m")
    
    # Add TMD as extra mass if enabled
    has_tmd = controls.damper_type == DamperType.TMD
    n_total = n + (1 if has_tmd else 0)
    nd = 3 * n_total
    
    M = np.zeros((nd, nd))
    K = np.zeros((nd, nd))
    
    # Rotational inertia (rectangular floor plate)
    J = m * (w**2 + d**2) / 12.0
    
    # Base isolation stiffness
    if controls.base_isolated:
        T_iso = controls.isolation_period
        k_iso = m * n * (2 * np.pi / T_iso)**2
    else:
        k_iso = k_col * 10  # Very stiff if not isolated
    
    for i in range(n):
        idx = 3 * i
        
        # Mass matrix (lumped)
        M[idx, idx] = m
        M[idx + 1, idx + 1] = m
        M[idx + 2, idx + 2] = J
        
        # Stiffness assembly
        if i == 0:
            # Base level
            kx = k_iso if controls.base_isolated else k_col
            ky = kx
        else:
            kx = k_col
            ky = k_col
        
        # Torsional stiffness (approximation)
        kt = k_col * (w**2 + d**2) / 24.0
        
        # Coupled stiffness matrix for one floor (including eccentricity)
        k_block = np.array([
            [kx, 0, -kx * ecc],
            [0, ky, ky * ecc * 0.5],
            [-kx * ecc, ky * ecc * 0.5, kt + kx * ecc**2 + ky * (ecc * 0.5)**2]
        ])
        
        # Add to global stiffness
        K[idx:idx + 3, idx:idx + 3] += k_block
        
        if i > 0:
            K[idx:idx + 3, idx - 3:idx] -= k_block
            K[idx - 3:idx, idx:idx + 3] -= k_block
            K[idx - 3:idx, idx - 3:idx] += k_block
    
    # Add TMD
    if has_tmd:
        tmd_mass = m * controls.tmd_mass_ratio * n
        tmd_floor = n - 1 if controls.tmd_floor < 0 else min(controls.tmd_floor, n - 1)
        
        # TMD DOFs start here
        idx_tmd = 3 * n
        idx_floor = 3 * tmd_floor
        
        M[idx_tmd, idx_tmd] = tmd_mass
        M[idx_tmd + 1, idx_tmd + 1] = tmd_mass
        M[idx_tmd + 2, idx_tmd + 2] = tmd_mass * 0.1  # Small rotational
        
        # TMD tuned to fundamental frequency (approximate)
        omega_struct = np.sqrt(k_col / m)
        k_tmd = tmd_mass * omega_struct**2
        
        # Connect TMD to floor
        K[idx_tmd, idx_tmd] = k_tmd
        K[idx_tmd, idx_floor] = -k_tmd
        K[idx_floor, idx_tmd] = -k_tmd
        K[idx_floor, idx_floor] += k_tmd
        
        K[idx_tmd + 1, idx_tmd + 1] = k_tmd
        K[idx_tmd + 1, idx_floor + 1] = -k_tmd
        K[idx_floor + 1, idx_tmd + 1] = -k_tmd
        K[idx_floor + 1, idx_floor + 1] += k_tmd
        
        # Add small rotational stiffness to prevent singular matrix
        # (TMD rotation DOF would otherwise have zero stiffness)
        K[idx_tmd + 2, idx_tmd + 2] = 1.0  # Dummy stiffness for numerical stability
    
    # Initial damping matrix (zeros - will be computed with Rayleigh)
    C = np.zeros((nd, nd))
    
    return M, K, C


def compute_rayleigh_damping(
    M: np.ndarray,
    K: np.ndarray,
    damping_ratio: float,
    controls: ControlDevices
) -> np.ndarray:
    """
    Compute Rayleigh damping matrix targeting first two modes.
    Includes additional damping for control devices.
    
    C = α·M + β·K
    
    where α and β are chosen to give the target damping ratio
    at the first two natural frequencies.
    
    Parameters
    ----------
    M : np.ndarray
        Mass matrix
    K : np.ndarray
        Stiffness matrix
    damping_ratio : float
        Target damping ratio (0 to 1)
    controls : ControlDevices
        Control device configuration
    
    Returns
    -------
    C : np.ndarray
        Damping matrix
    
    Notes
    -----
    If eigenvalue computation fails, uses conservative fallback frequencies.
    """
    # Validate damping ratio
    damping_ratio = max(0.0, min(damping_ratio, 1.0))
    
    # Eigen analysis for natural frequencies
    try:
        eig_vals, _ = eigh(K, M)
        # Filter out negative/zero eigenvalues (can occur with singular matrices)
        eig_vals = np.maximum(eig_vals, 1e-10)
        omega = np.sqrt(eig_vals)
        logger.debug(f"Modal frequencies: w1={omega[0]:.3f}, w2={omega[min(2, len(omega)-1)]:.3f} rad/s")
    except Exception as e:
        logger.warning(f"Eigenvalue computation failed: {e}. Using fallback frequencies.")
        omega = np.array([2 * np.pi, 2 * np.pi * 3])  # ~1Hz and ~3Hz
    
    w1 = omega[0]
    w2 = omega[min(2, len(omega) - 1)]
    
    # Rayleigh coefficients
    # ζ = (α/2ω + βω/2)
    # For ζ1 = ζ2 = ζ at ω1 and ω2:
    alpha = damping_ratio * 2 * w1 * w2 / (w1 + w2)
    beta = damping_ratio * 2 / (w1 + w2)
    
    C = alpha * M + beta * K
    
    # Additional damping from base isolation
    if controls.base_isolated:
        zeta_iso = controls.isolation_damping
        c_iso = 2 * zeta_iso * np.sqrt(K[0, 0] * M[0, 0])
        C[0, 0] += c_iso
        C[1, 1] += c_iso
    
    # Additional viscous dampers
    if controls.damper_type == DamperType.VISCOUS:
        nd = M.shape[0]
        n_floors = nd // 3
        for i in range(n_floors):
            idx = 3 * i
            C[idx, idx] += controls.viscous_c
            if i > 0:
                C[idx, idx - 3] -= controls.viscous_c * 0.5
                C[idx - 3, idx] -= controls.viscous_c * 0.5
                C[idx - 3, idx - 3] += controls.viscous_c * 0.5
    
    # Validate result
    if np.any(np.isnan(C)) or np.any(np.isinf(C)):
        logger.error("NaN/Inf in damping matrix")
        raise MatrixError("construction", "Damping matrix", "Contains NaN/Inf values")
    
    return C


def perform_modal_analysis(
    M: np.ndarray,
    K: np.ndarray,
    num_modes: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform modal analysis to obtain natural periods and mode shapes.
    
    Parameters
    ----------
    M : np.ndarray
        Mass matrix
    K : np.ndarray
        Stiffness matrix
    num_modes : int, optional
        Number of modes to return (default: all)
    
    Returns
    -------
    periods : np.ndarray
        Natural periods in seconds
    mode_shapes : np.ndarray
        Mode shape matrix (columns are mode shapes)
    
    Raises
    ------
    MatrixError
        If eigenvalue decomposition fails
    """
    nd = M.shape[0]
    
    try:
        eig_vals, eig_vecs = eigh(K, M)
        
        # Filter out negative eigenvalues (indicates numerical issues)
        eig_vals = np.maximum(eig_vals, 1e-10)
        omega = np.sqrt(eig_vals)
        
        # Convert to periods, avoiding division by zero
        periods = np.where(omega > 1e-10, 2 * np.pi / omega, 100.0)
        
        # Limit number of modes if specified
        if num_modes is not None and num_modes < nd:
            periods = periods[:num_modes]
            eig_vecs = eig_vecs[:, :num_modes]
        
        logger.debug(f"Modal analysis complete. T1={periods[0]:.4f}s, T2={periods[min(1, len(periods)-1)]:.4f}s")
        
        return periods, eig_vecs
        
    except np.linalg.LinAlgError as e:
        logger.error(f"Modal analysis failed: {e}")
        # Return fallback values
        periods = np.ones(nd) * 1.0
        mode_shapes = np.eye(nd)
        logger.warning("Using fallback modal properties (T=1.0s, identity modes)")
        return periods, mode_shapes
    except Exception as e:
        logger.error(f"Unexpected error in modal analysis: {e}")
        nd = M.shape[0]
        periods = np.ones(nd) * 1.0
        mode_shapes = np.eye(nd)
        return periods, mode_shapes


def compute_modal_participation(
    M: np.ndarray,
    mode_shapes: np.ndarray,
    direction: str = "X"
) -> np.ndarray:
    """
    Compute modal participation factors.
    
    Γ_n = (φ_n^T · M · r) / (φ_n^T · M · φ_n)
    
    Parameters
    ----------
    M : np.ndarray
        Mass matrix
    mode_shapes : np.ndarray
        Mode shape matrix
    direction : str
        Direction: "X", "Y", or "XY"
    
    Returns
    -------
    gamma : np.ndarray
        Modal participation factors
    """
    nd = M.shape[0]
    
    # Influence vector
    r = np.zeros(nd)
    if "X" in direction:
        r[0::3] = 1.0
    if "Y" in direction:
        r[1::3] = 1.0
    
    n_modes = mode_shapes.shape[1]
    gamma = np.zeros(n_modes)
    
    for i in range(n_modes):
        phi = mode_shapes[:, i]
        num = phi @ M @ r
        den = phi @ M @ phi
        gamma[i] = num / den if den > 0 else 0
    
    return gamma


def compute_effective_modal_mass(
    M: np.ndarray,
    mode_shapes: np.ndarray,
    gamma: np.ndarray
) -> np.ndarray:
    """
    Compute effective modal mass for each mode.
    
    M_eff,n = Γ_n² · (φ_n^T · M · φ_n)
    
    Parameters
    ----------
    M : np.ndarray
        Mass matrix
    mode_shapes : np.ndarray
        Mode shape matrix
    gamma : np.ndarray
        Modal participation factors
    
    Returns
    -------
    m_eff : np.ndarray
        Effective modal masses
    """
    n_modes = mode_shapes.shape[1]
    m_eff = np.zeros(n_modes)
    
    for i in range(n_modes):
        phi = mode_shapes[:, i]
        m_eff[i] = gamma[i]**2 * (phi @ M @ phi)
    
    return m_eff
