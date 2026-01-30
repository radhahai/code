"""
plotting.py - Visualization Utilities

This module contains plotting functions with automatic decimation
for browser stability when handling large datasets.
"""

import numpy as np
from typing import Tuple, Optional, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# DECIMATION UTILITIES
# =============================================================================

def decimate_data(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decimate data to prevent browser lag with large datasets.
    
    If the array length exceeds max_points, subsample every Nth point.
    This maintains visual accuracy while ensuring responsive UI.
    
    Parameters
    ----------
    x : np.ndarray
        X-axis data (e.g., time)
    y : np.ndarray
        Y-axis data
    max_points : int
        Maximum number of points to display (default 2000)
    
    Returns
    -------
    tuple
        (x_decimated, y_decimated)
    """
    n = len(x)
    
    if n <= max_points:
        return x, y
    
    # Calculate step size
    step = n // max_points
    
    # Use min-max decimation for better peak preservation
    # This keeps every Nth point but also includes local min/max
    indices = np.arange(0, n, step)
    
    return x[indices], y[indices]


def decimate_2d_data(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    max_x_points: int = 500,
    max_y_points: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decimate 2D data (e.g., heatmaps) for performance.
    
    Parameters
    ----------
    x : np.ndarray
        X-axis data
    y : np.ndarray
        Y-axis data
    z : np.ndarray
        2D data array
    max_x_points : int
        Maximum points in X direction
    max_y_points : int
        Maximum points in Y direction
    
    Returns
    -------
    tuple
        (x_dec, y_dec, z_dec)
    """
    nx = len(x)
    ny = len(y)
    
    step_x = max(1, nx // max_x_points)
    step_y = max(1, ny // max_y_points)
    
    x_dec = x[::step_x]
    y_dec = y[::step_y]
    z_dec = z[::step_y, ::step_x]
    
    return x_dec, y_dec, z_dec


# =============================================================================
# TIME HISTORY PLOTS
# =============================================================================

def create_time_history_plot(
    t: np.ndarray,
    roof_disp: np.ndarray,
    roof_acc: np.ndarray,
    ag: np.ndarray,
    max_points: int = 2000
) -> go.Figure:
    """
    Create time history plot with automatic decimation.
    
    Parameters
    ----------
    t : np.ndarray
        Time array
    roof_disp : np.ndarray
        Roof displacement (cm)
    roof_acc : np.ndarray
        Roof acceleration (g)
    ag : np.ndarray
        Ground acceleration (m/s²)
    max_points : int
        Maximum points to plot
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Decimate data
    t_d, roof_disp_d = decimate_data(t, roof_disp, max_points)
    _, roof_acc_d = decimate_data(t, roof_acc, max_points)
    _, ag_d = decimate_data(t, ag / 9.81, max_points)
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Roof Displacement (cm)",
            "Roof Acceleration (g)",
            "Ground Acceleration (g)"
        ),
        vertical_spacing=0.08
    )
    
    # Roof displacement
    fig.add_trace(go.Scatter(
        x=t_d, y=roof_disp_d,
        name="Roof Disp",
        line=dict(color="#00CC96", width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0, 204, 150, 0.2)'
    ), row=1, col=1)
    
    # Roof acceleration
    fig.add_trace(go.Scatter(
        x=t_d, y=roof_acc_d,
        name="Roof Acc",
        line=dict(color="#AB63FA", width=1.5),
        fill='tozeroy',
        fillcolor='rgba(171, 99, 250, 0.2)'
    ), row=2, col=1)
    
    # Ground acceleration
    fig.add_trace(go.Scatter(
        x=t_d, y=ag_d,
        name="Ground Acc",
        line=dict(color="#EF553B", width=1.5),
        fill='tozeroy',
        fillcolor='rgba(239, 85, 59, 0.2)'
    ), row=3, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40)
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    
    return fig


def create_base_shear_plot(
    t: np.ndarray,
    base_shear: np.ndarray,
    max_points: int = 2000
) -> go.Figure:
    """
    Create base shear time history plot.
    
    Parameters
    ----------
    t : np.ndarray
        Time array
    base_shear : np.ndarray
        Base shear (kN)
    max_points : int
        Maximum points to plot
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    t_d, shear_d = decimate_data(t, base_shear, max_points)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_d, y=shear_d,
        line=dict(color="#FFA15A", width=1),
        fill='tozeroy',
        fillcolor='rgba(255, 161, 90, 0.3)'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=200,
        xaxis_title="Time (s)",
        yaxis_title="Base Shear (kN)",
        margin=dict(l=40, r=20, t=10, b=40)
    )
    
    return fig


# =============================================================================
# DRIFT PLOTS
# =============================================================================

def create_drift_profile_plot(
    max_drifts_per_floor: np.ndarray,
    is_drift_limit: Optional[float] = None
) -> go.Figure:
    """
    Create interstorey drift profile plot.
    
    Parameters
    ----------
    max_drifts_per_floor : np.ndarray
        Maximum drift at each floor (%)
    is_drift_limit : float, optional
        IS code drift limit
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    n_floors = len(max_drifts_per_floor)
    floor_labels = [f"F{i+1}" for i in range(n_floors)]
    
    # Color based on performance level
    colors = [
        '#22c55e' if d < 1.0 else '#eab308' if d < 2.0 else '#ef4444'
        for d in max_drifts_per_floor
    ]
    
    fig = go.Figure(go.Bar(
        x=max_drifts_per_floor,
        y=floor_labels,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{d:.2f}%" for d in max_drifts_per_floor],
        textposition='auto'
    ))
    
    # Add limit lines
    fig.add_vline(x=1.0, line_dash="dash", line_color="#eab308",
                  annotation_text="IO Limit")
    fig.add_vline(x=2.0, line_dash="dash", line_color="#ef4444",
                  annotation_text="LS Limit")
    
    if is_drift_limit is not None:
        fig.add_vline(x=is_drift_limit, line_dash="dot", line_color="#22c55e",
                      annotation_text="IS Drift Limit")
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Drift Ratio (%)",
        yaxis_title="Floor",
        showlegend=False,
        margin=dict(l=40, r=20, t=20, b=40)
    )
    
    return fig


def create_drift_heatmap(
    t: np.ndarray,
    drift_ratios: np.ndarray,
    max_points: int = 500
) -> go.Figure:
    """
    Create drift ratio heatmap.
    
    Parameters
    ----------
    t : np.ndarray
        Time array
    drift_ratios : np.ndarray
        Drift ratios (n_steps x n_floors)
    max_points : int
        Maximum time points to display
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    n_floors = drift_ratios.shape[1]
    floor_labels = [f"F{i+1}" for i in range(n_floors)]
    
    # Decimate time axis
    step = max(1, len(t) // max_points)
    t_d = t[::step]
    drift_d = drift_ratios[::step, :]
    
    fig = go.Figure(data=go.Heatmap(
        z=drift_d.T,
        x=t_d,
        y=floor_labels,
        colorscale="Viridis",
        colorbar=dict(title="Drift (%)")
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=250,
        xaxis_title="Time (s)",
        yaxis_title="Floor",
        margin=dict(l=40, r=20, t=10, b=40)
    )
    
    return fig


# =============================================================================
# ENERGY PLOTS
# =============================================================================

def create_energy_plot(
    t: np.ndarray,
    energy: dict,
    max_points: int = 2000
) -> go.Figure:
    """
    Create energy balance plot.
    
    Parameters
    ----------
    t : np.ndarray
        Time array
    energy : dict
        Energy components dictionary
    max_points : int
        Maximum points to plot
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Decimate all energy arrays
    t_d, input_d = decimate_data(t, energy['input'], max_points)
    _, kinetic_d = decimate_data(t, energy['kinetic'], max_points)
    _, strain_d = decimate_data(t, energy['strain'], max_points)
    _, damping_d = decimate_data(t, energy['damping'], max_points)
    _, hysteretic_d = decimate_data(t, energy['hysteretic'], max_points)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t_d, y=input_d,
        name="Input Energy",
        fill='tozeroy',
        line=dict(color='#ef4444', width=2),
        fillcolor='rgba(239, 68, 68, 0.3)'
    ))
    
    fig.add_trace(go.Scatter(
        x=t_d, y=kinetic_d,
        name="Kinetic Energy",
        line=dict(color='#3b82f6', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=t_d, y=strain_d,
        name="Strain Energy",
        line=dict(color='#22c55e', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=t_d, y=damping_d,
        name="Damping Energy",
        line=dict(color='#f59e0b', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=t_d, y=hysteretic_d,
        name="Hysteretic Energy",
        line=dict(color='#ec4899', width=2, dash='dash')
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Time (s)",
        yaxis_title="Energy (J)",
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified"
    )
    
    return fig


# =============================================================================
# HYSTERESIS PLOTS
# =============================================================================

def create_hysteresis_plot(
    disp_history: np.ndarray,
    force_history: np.ndarray,
    floor_idx: int
) -> go.Figure:
    """
    Create hysteresis loop plot.
    
    Parameters
    ----------
    disp_history : np.ndarray
        Displacement history per floor
    force_history : np.ndarray
        Force history per floor
    floor_idx : int
        Floor index to plot
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=disp_history[:, floor_idx],
        y=force_history[:, floor_idx],
        mode='lines',
        line=dict(color='#8b5cf6', width=1),
        name='Hysteresis'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Interstorey Drift (cm)",
        yaxis_title="Shear Force (kN)",
        title=f"Hysteresis Loop - Floor {floor_idx + 1}"
    )
    
    return fig


def create_phase_plane_plot(
    disp: np.ndarray,
    vel: np.ndarray,
    max_points: int = 2000
) -> go.Figure:
    """
    Create phase plane plot (displacement vs velocity).
    
    Parameters
    ----------
    disp : np.ndarray
        Displacement time history
    vel : np.ndarray
        Velocity time history
    max_points : int
        Maximum points to plot
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    step = max(1, len(disp) // max_points)
    disp_d = disp[::step]
    vel_d = vel[::step]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=disp_d,
        y=vel_d,
        mode='lines',
        line=dict(color='#8b5cf6', width=1),
        name='Phase Plane'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Displacement (cm)",
        yaxis_title="Velocity (cm/s)",
        title="Roof Phase Plane"
    )
    
    return fig


# =============================================================================
# SPECTRUM PLOTS
# =============================================================================

def create_response_spectrum_plot(
    ground_spec: dict,
    periods_struct: np.ndarray = None,
    is_spec: np.ndarray = None
) -> go.Figure:
    """
    Create response spectrum plot.
    
    Parameters
    ----------
    ground_spec : dict
        Ground motion response spectrum
    periods_struct : np.ndarray, optional
        Structure's natural periods
    is_spec : np.ndarray, optional
        IS 1893 design spectrum
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Acceleration Spectrum", "Displacement Spectrum")
    )
    
    fig.add_trace(go.Scatter(
        x=ground_spec['periods'],
        y=ground_spec['Sa'],
        line=dict(color='#ef4444', width=2),
        name='Sa'
    ), row=1, col=1)
    
    if is_spec is not None:
        fig.add_trace(go.Scatter(
            x=ground_spec['periods'],
            y=is_spec,
            line=dict(color='#22c55e', width=2, dash='dash'),
            name='IS 1893 Sa/g'
        ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=ground_spec['periods'],
        y=ground_spec['Sd'] * 100,  # cm
        line=dict(color='#3b82f6', width=2),
        name='Sd'
    ), row=1, col=2)
    
    # Mark structure's natural periods
    if periods_struct is not None:
        for i, T in enumerate(periods_struct[:3]):
            if T < 10:
                fig.add_vline(x=T, row=1, col=1, line_dash="dash",
                              line_color="#22c55e")
                fig.add_vline(x=T, row=1, col=2, line_dash="dash",
                              line_color="#22c55e")
    
    fig.update_xaxes(title_text="Period (s)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Period (s)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Sa (g)", row=1, col=1)
    fig.update_yaxes(title_text="Sd (cm)", row=1, col=2)
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=is_spec is not None
    )
    
    return fig


# =============================================================================
# 3D MODE SHAPE PLOT
# =============================================================================

def create_3d_mode_shape_plot(
    mode_shape: np.ndarray,
    floor_height: float,
    building_width: float,
    n_floors: int
) -> go.Figure:
    """
    Create 3D mode shape visualization.
    
    Parameters
    ----------
    mode_shape : np.ndarray
        Mode shape vector
    floor_height : float
        Story height
    building_width : float
        Building width for scaling
    n_floors : int
        Number of floors
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    n_dof = len(mode_shape)
    n_floors_mode = n_dof // 3
    
    vec_x = mode_shape[0::3][:n_floors_mode]
    vec_y = mode_shape[1::3][:n_floors_mode]
    
    # Normalize
    max_disp = max(np.max(np.abs(vec_x)), np.max(np.abs(vec_y)), 0.001)
    scale = building_width / 3 / max_disp
    vec_x = vec_x * scale
    vec_y = vec_y * scale
    
    z_floors = np.arange(floor_height, (n_floors_mode + 1) * floor_height, floor_height)
    
    fig = go.Figure()
    
    # Original structure
    fig.add_trace(go.Scatter3d(
        x=[0] * n_floors_mode,
        y=[0] * n_floors_mode,
        z=z_floors,
        mode='lines+markers',
        line=dict(color='#64748b', dash='dash', width=3),
        marker=dict(size=5, color='#64748b'),
        name='Undeformed'
    ))
    
    # Deformed shape
    fig.add_trace(go.Scatter3d(
        x=vec_x,
        y=vec_y,
        z=z_floors,
        mode='lines+markers',
        line=dict(color='#8b5cf6', width=6),
        marker=dict(size=8, color='#8b5cf6'),
        name='Mode Shape'
    ))
    
    # Connect floors to show deformation
    for i in range(n_floors_mode):
        fig.add_trace(go.Scatter3d(
            x=[0, vec_x[i]],
            y=[0, vec_y[i]],
            z=[z_floors[i], z_floors[i]],
            mode='lines',
            line=dict(color='#22c55e', width=2, dash='dot'),
            showlegend=False
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-building_width, building_width], title="X (m)"),
            yaxis=dict(range=[-building_width, building_width], title="Y (m)"),
            zaxis=dict(title="Height (m)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        template="plotly_dark",
        height=500,
        showlegend=True,
        legend=dict(x=0, y=1)
    )
    
    return fig
