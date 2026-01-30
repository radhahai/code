"""
solvers.py - Time History Analysis Solvers

This module contains the numerical solvers for dynamic analysis:
- Newmark-Beta linear solver
- Newmark-Beta nonlinear solver with hysteresis

All solvers are optimized for performance using NumPy vectorization.
Includes comprehensive error handling and numerical stability checks.
"""

import numpy as np
from typing import Dict, Any, Union, Tuple, Optional
from scipy.linalg import solve, inv
import streamlit as st
import warnings

from exceptions import SolverError, NumericalInstabilityError, MatrixError
from logging_config import get_logger

logger = get_logger("solvers")

# Numerical stability constants
EPSILON = 1e-12
MAX_CONDITION_NUMBER = 1e15
MAX_DISPLACEMENT = 1e6  # meters - unrealistic threshold

# =============================================================================
# LINEAR SOLVER
# =============================================================================

@st.cache_data(show_spinner=False)
def newmark_linear_solver_cached(
    M_bytes: bytes,
    K_bytes: bytes,
    C_bytes: bytes,
    ag_bytes: bytes,
    t_bytes: bytes,
    M_shape: Tuple[int, int],
    ag_shape: Tuple[int],
    direction: str
) -> Dict[str, np.ndarray]:
    """Cached wrapper for linear solver."""
    M = np.frombuffer(M_bytes).reshape(M_shape)
    K = np.frombuffer(K_bytes).reshape(M_shape)
    C = np.frombuffer(C_bytes).reshape(M_shape)
    ag = np.frombuffer(ag_bytes).reshape(ag_shape)
    t = np.frombuffer(t_bytes)
    
    return _newmark_linear_solver_impl(M, K, C, ag, t, direction)


def newmark_linear_solver(
    M: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    ag: Union[np.ndarray, Dict[str, np.ndarray]],
    t: np.ndarray,
    direction: str = "X"
) -> Dict[str, np.ndarray]:
    """
    Newmark-Beta Integration (Average Acceleration Method).
    
    Solves: M·ü + C·u̇ + K·u = -M·r·ag
    
    Parameters
    ----------
    M : np.ndarray
        Mass matrix
    K : np.ndarray
        Stiffness matrix
    C : np.ndarray
        Damping matrix
    ag : np.ndarray or dict
        Ground acceleration (m/s²) or dict with 'x' and 'y' components
    t : np.ndarray
        Time array
    direction : str
        Loading direction: "X", "Y", or "XY"
    
    Returns
    -------
    dict
        Results containing displacement, velocity, acceleration arrays
    """
    # Handle bidirectional input
    if isinstance(ag, dict):
        ag_x = ag.get("x", np.zeros_like(t))
        ag_y = ag.get("y", np.zeros_like(t))
    else:
        ag_x = ag
        ag_y = np.zeros_like(ag)
    
    return _newmark_linear_solver_impl(M, K, C, ag_x, t, direction, ag_y)


def _newmark_linear_solver_impl(
    M: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    ag_x: np.ndarray,
    t: np.ndarray,
    direction: str,
    ag_y: np.ndarray = None
) -> Dict[str, np.ndarray]:
    """Implementation of Newmark-Beta linear solver with robustness checks."""
    
    if ag_y is None:
        ag_y = np.zeros_like(ag_x)
    
    dt = t[1] - t[0]
    num_dof = M.shape[0]
    n_steps = len(t)
    
    # Validate inputs
    if num_dof == 0:
        raise SolverError("newmark_linear", 0, "Zero degrees of freedom")
    
    if n_steps < 2:
        raise SolverError("newmark_linear", 0, "Insufficient time steps")
    
    logger.debug(f"Linear solver: {num_dof} DOFs, {n_steps} steps, dt={dt:.4f}s")
    
    # Influence vectors
    r_x = np.zeros(num_dof)
    r_y = np.zeros(num_dof)
    
    if "X" in direction:
        r_x[0::3] = 1.0
    if "Y" in direction:
        r_y[1::3] = 1.0
    
    # Initialize arrays
    u = np.zeros((n_steps, num_dof))
    v = np.zeros((n_steps, num_dof))
    a = np.zeros((n_steps, num_dof))
    
    # Initial conditions
    p0 = -M @ (r_x * ag_x[0] + r_y * ag_y[0])
    try:
        a[0] = solve(M, p0 - C @ v[0] - K @ u[0])
    except np.linalg.LinAlgError as e:
        logger.warning(f"Mass matrix solve failed, using lstsq: {e}")
        a[0] = np.linalg.lstsq(M, p0 - C @ v[0] - K @ u[0], rcond=None)[0]
    
    # Newmark constants (Average Acceleration Method)
    gamma = 0.5
    beta = 0.25
    
    a1 = 1.0 / (beta * dt**2)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2 * beta) - 1.0
    a4 = gamma / (beta * dt)
    a5 = gamma / beta - 1
    a6 = dt * (gamma / (2 * beta) - 1)
    
    # Effective stiffness matrix
    K_eff = K + a1 * M + a4 * C
    
    # Check condition number for numerical stability
    try:
        cond_num = np.linalg.cond(K_eff)
        if cond_num > MAX_CONDITION_NUMBER:
            logger.warning(f"K_eff poorly conditioned: {cond_num:.2e}")
    except Exception:
        pass
    
    try:
        K_inv = inv(K_eff)
    except np.linalg.LinAlgError as e:
        logger.warning(f"K_eff inversion failed, using pseudo-inverse: {e}")
        K_inv = np.linalg.pinv(K_eff)
    
    # Time stepping with stability monitoring
    instability_detected = False
    for i in range(n_steps - 1):
        # Effective load
        p_eff = (-M @ (r_x * ag_x[i + 1] + r_y * ag_y[i + 1]) +
                 M @ (a1 * u[i] + a2 * v[i] + a3 * a[i]) +
                 C @ (a4 * u[i] + a5 * v[i] + a6 * a[i]))
        
        u[i + 1] = K_inv @ p_eff
        v[i + 1] = a4 * (u[i + 1] - u[i]) - a5 * v[i] - a6 * a[i]
        a[i + 1] = a1 * (u[i + 1] - u[i]) - a2 * v[i] - a3 * a[i]
        
        # Check for numerical instability every 100 steps
        if i % 100 == 0 and not instability_detected:
            if np.any(np.isnan(u[i + 1])) or np.any(np.isinf(u[i + 1])):
                logger.error(f"NaN/Inf detected at step {i}")
                instability_detected = True
            elif np.max(np.abs(u[i + 1])) > MAX_DISPLACEMENT:
                logger.warning(f"Unrealistic displacement at step {i}: {np.max(np.abs(u[i + 1])):.2e} m")
                instability_detected = True
    
    # Final stability check
    if np.any(np.isnan(u)) or np.any(np.isinf(u)):
        raise NumericalInstabilityError("newmark_linear_solver", "displacement")
    
    # Absolute acceleration
    a_abs = a + np.outer(ag_x, r_x) + np.outer(ag_y, r_y)
    
    logger.debug(f"Linear solver complete. Max disp: {np.max(np.abs(u)):.4f} m")
    
    return {
        "displacement": u,
        "velocity": v,
        "relative_acceleration": a,
        "absolute_acceleration": a_abs,
        "influence_vector": r_x + r_y,
        "ag_x": ag_x,
        "ag_y": ag_y
    }


# =============================================================================
# NONLINEAR SOLVER
# =============================================================================

@st.cache_data(show_spinner=False)
def newmark_nonlinear_solver_cached(
    M_bytes: bytes,
    K_bytes: bytes,
    C_bytes: bytes,
    ag_bytes: bytes,
    t_bytes: bytes,
    M_shape: Tuple[int, int],
    ag_shape: Tuple[int],
    yield_drift: float,
    hardening_ratio: float,
    floor_height: float
) -> Dict[str, Any]:
    """Cached wrapper for nonlinear solver."""
    M = np.frombuffer(M_bytes).reshape(M_shape)
    K = np.frombuffer(K_bytes).reshape(M_shape)
    C = np.frombuffer(C_bytes).reshape(M_shape)
    ag = np.frombuffer(ag_bytes).reshape(ag_shape)
    t = np.frombuffer(t_bytes)
    
    return newmark_nonlinear_solver(M, K, C, ag, t, yield_drift, hardening_ratio, floor_height)


def newmark_nonlinear_solver(
    M: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    ag: np.ndarray,
    t: np.ndarray,
    yield_drift: float,
    hardening_ratio: float,
    floor_height: float
) -> Dict[str, Any]:
    """
    Nonlinear Newmark-Beta with bilinear hysteresis.
    
    IMPORTANT: This solver currently only supports X-direction (1D) analysis.
    Bidirectional loading is NOT supported and will produce incorrect results.
    The hysteresis model operates on X-direction drift only.
    
    Note on Physics: This simplified model treats each floor as an isolated
    1D spring for nonlinear behavior. Torsional coupling effects from 
    eccentricity (as modeled in structural.py) are handled linearly during
    the plastic phase. This is a simplification suitable for educational
    purposes but may not capture full 3D nonlinear behavior.
    
    Parameters
    ----------
    M : np.ndarray
        Mass matrix
    K : np.ndarray
        Stiffness matrix
    C : np.ndarray
        Damping matrix
    ag : np.ndarray
        Ground acceleration in X-direction (m/s²)
    t : np.ndarray
        Time array
    yield_drift : float
        Yield drift ratio
    hardening_ratio : float
        Post-yield stiffness ratio (α)
    floor_height : float
        Story height in meters
    
    Returns
    -------
    dict
        Results including displacement, hysteresis, ductility, residual drift
    """
    dt = t[1] - t[0]
    num_dof = M.shape[0]
    n_floors = num_dof // 3
    n_steps = len(t)
    
    # Validate inputs
    if n_floors < 1:
        raise SolverError("newmark_nonlinear", 0, "No floors detected")
    if yield_drift <= 0:
        raise SolverError("newmark_nonlinear", 0, "yield_drift must be positive")
    
    logger.debug(f"Nonlinear solver: {n_floors} floors, {n_steps} steps, yield_drift={yield_drift:.4f}")
    
    # X-direction influence vector only
    r = np.zeros(num_dof)
    r[0::3] = 1.0
    
    # Initialize arrays
    u = np.zeros((n_steps, num_dof))
    v = np.zeros((n_steps, num_dof))
    a = np.zeros((n_steps, num_dof))
    
    # Hysteresis state variables
    u_yield = yield_drift * floor_height
    alpha = max(hardening_ratio, 0.001)  # Ensure minimum hardening for stability
    
    # State for each floor X direction
    u_plastic = np.zeros(n_floors)
    yielded = np.zeros(n_floors, dtype=bool)
    max_ductility = np.zeros(n_floors)
    force_history = np.zeros((n_steps, n_floors))
    disp_history = np.zeros((n_steps, n_floors))
    
    # Initial acceleration
    p0 = -M @ r * ag[0]
    try:
        a[0] = solve(M, p0)
    except np.linalg.LinAlgError:
        logger.warning("Initial acceleration solve failed, using zeros")
        a[0] = np.zeros(num_dof)
    
    # Newmark parameters
    gamma = 0.5
    beta = 0.25
    
    K_orig = K.copy()
    K_tangent = K.copy()
    
    # Track yielding events for logging
    yield_events = 0
    
    # Time stepping with Newton-Raphson iteration
    for i in range(n_steps - 1):
        # Predictor
        u_pred = u[i] + dt * v[i] + (0.5 - beta) * dt**2 * a[i]
        v_pred = v[i] + (1 - gamma) * dt * a[i]
        
        # Initialize internal force vector for nonlinear computation
        f_int = np.zeros(num_dof)
        
        # Compute nonlinear internal forces from floor hysteresis models
        floor_forces = np.zeros(n_floors)  # Store floor forces for f_int assembly
        
        # Update hysteresis for each floor
        for fl in range(n_floors):
            idx = 3 * fl
            
            # Interstorey drift
            if fl == 0:
                drift = u_pred[idx]
            else:
                drift = u_pred[idx] - u_pred[idx - 3]
            
            disp_history[i + 1, fl] = drift * 100  # cm
            
            # Check yielding
            drift_rel = drift - u_plastic[fl]
            force = K_orig[idx, idx] * drift_rel
            
            if abs(drift_rel) > u_yield and not yielded[fl]:
                yielded[fl] = True
            
            if yielded[fl]:
                # Post-yield stiffness (bilinear hysteresis)
                if abs(drift_rel) > u_yield:
                    sign = np.sign(drift_rel)
                    force = (sign * K_orig[idx, idx] * u_yield +
                            alpha * K_orig[idx, idx] * (drift_rel - sign * u_yield))
                    u_plastic[fl] = drift - sign * u_yield * (1 + alpha * (abs(drift_rel) / u_yield - 1))
            
            force_history[i + 1, fl] = force / 1000  # kN
            max_ductility[fl] = max(max_ductility[fl], abs(drift_rel) / u_yield)
            floor_forces[fl] = force
        
        # Assemble f_int from nonlinear floor forces (shear building model)
        # Each floor restoring force acts on that floor and reacts on floor below
        for fl in range(n_floors):
            idx = 3 * fl
            # Force from this floor's spring
            f_int[idx] += floor_forces[fl]
            # Reaction on floor below (coupling)
            if fl > 0:
                f_int[idx - 3] -= floor_forces[fl]
        
        # Add contributions from Y and rotational DOFs using linear stiffness
        # (nonlinearity only in X-direction shear for this simplified model)
        for fl in range(n_floors):
            idx = 3 * fl
            # Y-direction (linear)
            f_int[idx + 1] += K_orig[idx + 1, idx + 1] * u_pred[idx + 1]
            if fl > 0:
                f_int[idx + 1] -= K_orig[idx + 1, idx - 2] * u_pred[idx - 2]
                f_int[idx - 2] -= K_orig[idx - 2, idx + 1] * u_pred[idx + 1]
            # Rotation (linear)
            f_int[idx + 2] += K_orig[idx + 2, idx + 2] * u_pred[idx + 2]
            if fl > 0:
                f_int[idx + 2] -= K_orig[idx + 2, idx - 1] * u_pred[idx - 1]
                f_int[idx - 1] -= K_orig[idx - 1, idx + 2] * u_pred[idx + 2]
        
        # Newton-Raphson correction
        residual = M @ a[i] + C @ v_pred + f_int + M @ r * ag[i + 1]
        K_eff = K_tangent + gamma / (beta * dt) * C + 1 / (beta * dt**2) * M
        
        try:
            du = solve(K_eff, -residual)
        except np.linalg.LinAlgError:
            logger.warning(f"Newton-Raphson solve failed at step {i}, using zero correction")
            du = np.zeros(num_dof)
        
        u[i + 1] = u_pred + du
        a[i + 1] = (u[i + 1] - u[i] - dt * v[i]) / (beta * dt**2) - (0.5 / beta - 1) * a[i]
        v[i + 1] = v[i] + dt * ((1 - gamma) * a[i] + gamma * a[i + 1])
        
        # Stability check every 100 steps
        if i % 100 == 0:
            if np.any(np.isnan(u[i + 1])) or np.any(np.isinf(u[i + 1])):
                raise NumericalInstabilityError("newmark_nonlinear_solver", "displacement")
            if np.max(np.abs(u[i + 1])) > MAX_DISPLACEMENT:
                logger.error(f"Unrealistic displacement at step {i}: {np.max(np.abs(u[i + 1])):.2e} m")
                raise NumericalInstabilityError("newmark_nonlinear_solver", "displacement (exceeded limit)")
    
    # Final validation
    if np.any(np.isnan(u)) or np.any(np.isinf(u)):
        raise NumericalInstabilityError("newmark_nonlinear_solver", "final displacement")
    
    # Absolute acceleration
    a_abs = a + np.outer(ag, r)
    
    # Count yielding events
    yield_count = np.sum(yielded)
    logger.debug(f"Nonlinear solver complete. Floors yielded: {yield_count}/{n_floors}")
    
    # Compute residual drift (permanent deformation after shaking stops)
    # Take average of last 10% of response as residual
    n_tail = max(int(0.1 * n_steps), 10)
    residual_drift = np.zeros(n_floors)
    
    for fl in range(n_floors):
        idx = 3 * fl
        if fl == 0:
            floor_drift = u[-n_tail:, idx]
        else:
            floor_drift = u[-n_tail:, idx] - u[-n_tail:, idx - 3]
        residual_drift[fl] = np.mean(floor_drift) / floor_height * 100  # percentage
    
    return {
        "displacement": u,
        "velocity": v,
        "relative_acceleration": a,
        "absolute_acceleration": a_abs,
        "influence_vector": r,
        "force_history": force_history,
        "disp_history": disp_history,
        "max_ductility": max_ductility,
        "yielded": yielded,
        "residual_drift": residual_drift,
        "u_plastic": u_plastic
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_base_shear(
    M: np.ndarray,
    a_abs: np.ndarray
) -> np.ndarray:
    """
    Compute base shear time history.
    
    V_base = Σ(m_i × a_i)
    
    Parameters
    ----------
    M : np.ndarray
        Mass matrix
    a_abs : np.ndarray
        Absolute acceleration array (n_steps x n_dof)
    
    Returns
    -------
    np.ndarray
        Base shear in kN
    """
    base_shear = np.sum(a_abs @ M, axis=1) / 1000  # kN
    return base_shear


def compute_interstorey_drifts(
    u: np.ndarray,
    floor_height: float,
    n_floors: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute interstorey drift ratios.
    
    Parameters
    ----------
    u : np.ndarray
        Displacement array (n_steps x n_dof)
    floor_height : float
        Story height in meters
    n_floors : int
        Number of floors
    
    Returns
    -------
    drift_ratios : np.ndarray
        Drift ratio time history (percentage)
    max_drifts_per_floor : np.ndarray
        Maximum drift at each floor
    max_drift : float
        Overall maximum drift ratio
    """
    # Extract X-direction displacements
    u_x = u[:, 0::3][:, :n_floors]
    
    # Add ground level (zero displacement)
    n_steps = u.shape[0]
    u_x_aug = np.column_stack([np.zeros(n_steps), u_x])
    
    # Interstorey drifts
    drifts = np.diff(u_x_aug, axis=1)
    drift_ratios = np.abs(drifts) / floor_height * 100  # percentage
    
    max_drifts_per_floor = np.max(drift_ratios, axis=0)
    max_drift = np.max(drift_ratios)
    
    return drift_ratios, max_drifts_per_floor, max_drift
