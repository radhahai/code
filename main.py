"""
main.py - Seismic Analysis Tool Main Application

Advanced MDOF Shear Building Simulation for Structural Engineering Education.

This is the main entry point for the Streamlit application.
Run with: streamlit run main.py

Author: Seismic Analysis Educational Tool
License: MIT
"""

import json
import sys
from datetime import datetime
from typing import Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

# Local imports - production utilities first
from logging_config import get_logger, PerformanceLogger
from exceptions import SeismicAppError, SolverError, ValidationError
from production_config import AppConfig

# Initialize logger and config
logger = get_logger("main")
app_config = AppConfig.from_environment()

from models import (
    SeismicEvent, DamperType, AnalysisType,
    StructuralProperties, ControlDevices
)
from config import (
    DEFAULT_CONFIG, ensure_config_state, apply_config, build_config_snapshot,
    compute_scaled_stiffness, check_period_warning
)
from styles import (
    APP_STYLES, APP_HEADER, APP_SUBTITLE, SYNTHETIC_DATA_DISCLAIMER,
    get_feature_card, WELCOME_FEATURES
)
from ground_motion import generate_ground_motion, generate_bidirectional_motion
from structural import (
    build_structural_matrices, compute_rayleigh_damping, perform_modal_analysis
)
from solvers import (
    newmark_linear_solver, newmark_nonlinear_solver,
    compute_base_shear, compute_interstorey_drifts
)
from postprocess import (
    calculate_energy_balance, compute_response_spectrum,
    compute_floor_response_spectrum, compute_fragility_curves,
    assess_performance_level, compute_damage_index,
    compute_residual_drift_assessment, generate_recommendations
)
from design_codes import (
    compute_is1893_spectrum, compute_design_base_shear, get_drift_limit_is1893
)
from plotting import (
    create_time_history_plot, create_base_shear_plot,
    create_drift_profile_plot, create_drift_heatmap,
    create_energy_plot, create_hysteresis_plot, create_phase_plane_plot,
    create_response_spectrum_plot, create_3d_mode_shape_plot,
    decimate_data
)
from report import generate_pdf_report


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Seismic Analysis Tool | MDOF Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply styles
st.markdown(APP_STYLES, unsafe_allow_html=True)

# Header
st.markdown(APP_HEADER, unsafe_allow_html=True)
st.markdown(APP_SUBTITLE, unsafe_allow_html=True)

# Initialize configuration
ensure_config_state()


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <h2 style="margin: 0; color: #e2e8f0;">Control Center</h2>
        <p style="color: #94a3b8; margin: 0;">Configure and run analyses</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Educational Disclaimer
    st.markdown(SYNTHETIC_DATA_DISCLAIMER, unsafe_allow_html=True)
    
    # Configuration management
    with st.expander("Configuration", expanded=False):
        config_text = st.text_area(
            "Paste configuration (JSON)", 
            height=120, 
            key="config_text"
        )
        if st.button("Apply pasted configuration", use_container_width=True):
            try:
                loaded = json.loads(config_text)
                apply_config(loaded)
                st.success("Configuration applied.")
                st.rerun()
            except Exception as exc:
                st.error(f"Invalid JSON: {exc}")
        
        config_snapshot = build_config_snapshot()
        st.download_button(
            label="Download configuration",
            data=json.dumps(config_snapshot, indent=2),
            file_name="seismic_config.json",
            mime="application/json",
        )
        if st.button("Reset configuration", use_container_width=True):
            apply_config(DEFAULT_CONFIG)
            st.rerun()
    
    # BUILDING PROPERTIES
    with st.expander("Structure Properties", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            n_floors = st.slider(
                "Floors", 2, 30, 
                st.session_state.n_floors, 
                help="Number of stories",
                key="n_floors"
            )
        with col2:
            floor_height = st.number_input(
                "Height (m)", 2.5, 5.0, 
                st.session_state.floor_height, 0.1,
                key="floor_height"
            )
        
        mass = st.number_input(
            "Floor Mass (tonnes)", 10, 500, 
            st.session_state.floor_mass_tonnes, step=5,
            key="floor_mass_tonnes"
        ) * 1000  # Convert to kg
        
        col3, col4 = st.columns(2)
        with col3:
            width = st.number_input(
                "Width (m)", 5.0, 40.0, 
                st.session_state.building_width, 1.0,
                key="building_width"
            )
        with col4:
            depth = st.number_input(
                "Depth (m)", 5.0, 40.0, 
                st.session_state.building_depth, 1.0,
                key="building_depth"
            )
        
        stiffness_factor = st.slider(
            "Stiffness Factor", 0.3, 3.0, 
            st.session_state.stiffness_factor, 0.1,
            help="Multiplier for column stiffness (auto-scaled based on floors)",
            key="stiffness_factor"
        )
        
        # Show estimated period based on settings
        k_estimate = compute_scaled_stiffness(n_floors, mass, stiffness_factor)
        T_estimate = 0.1 * n_floors / stiffness_factor
        st.caption(f"Est. Period: ~{T_estimate:.2f}s | Stiffness: ~{k_estimate/1e6:.1f} MN/m")
        
        st.markdown("**Advanced Structural**")
        ecc = st.slider(
            "Eccentricity (%)", 0, 30, 
            st.session_state.eccentricity_pct,
            help="Mass-stiffness eccentricity causing torsion",
            key="eccentricity_pct"
        )
        irregularity = st.selectbox(
            "Vertical Irregularity",
            ["Regular", "Soft Story", "Setback", "Mass Irregular"],
            index=["Regular", "Soft Story", "Setback", "Mass Irregular"].index(
                st.session_state.irregularity
            ),
            key="irregularity",
        )
    
    # SEISMIC INPUT
    with st.expander("Earthquake Input", expanded=True):
        event_names = [e.value for e in SeismicEvent]
        event_selected = st.selectbox(
            "Ground Motion",
            event_names,
            index=event_names.index(st.session_state.event_selected),
            key="event_selected",
            help="All motions are SYNTHETICALLY GENERATED"
        )
        event_type = [e for e in SeismicEvent if e.value == event_selected][0]
        
        # Show info about synthetic data
        st.info("⚠️ Ground motions are synthetically generated for educational simulation.", icon="ℹ️")
        
        col5, col6 = st.columns(2)
        with col5:
            pga = st.slider("PGA (g)", 0.05, 2.0, st.session_state.pga, 0.05, key="pga")
        with col6:
            duration = st.slider("Duration (s)", 10, 120, st.session_state.duration, key="duration")
        
        soil = st.selectbox(
            "Soil Type",
            ["soft", "medium", "stiff", "rock"],
            index=["soft", "medium", "stiff", "rock"].index(st.session_state.soil),
            key="soil",
        )
        
        # Get current analysis type to determine direction options
        current_analysis = st.session_state.get("analysis_selected", AnalysisType.LINEAR.value)
        is_nonlinear = "Nonlinear" in current_analysis
        
        if is_nonlinear:
            # Force X-direction only for nonlinear analysis
            st.warning("⚠️ Nonlinear analysis is restricted to planar (1D) response. Direction locked to X.", icon="⚠️")
            direction = "X"
            st.session_state.direction = "X"
        else:
            direction = st.radio(
                "Loading Direction",
                ["X", "Y", "XY (Bidirectional)"],
                index=["X", "Y", "XY (Bidirectional)"].index(st.session_state.direction),
                horizontal=True,
                key="direction",
            )
        
        scale_factor = st.slider(
            "Scale Factor", 0.5, 2.0,
            st.session_state.scale_factor, 0.1,
            help="Multiply ground motion amplitude",
            key="scale_factor",
        )
        
        st.markdown("**Reproducibility**")
        seed_enabled = st.checkbox(
            "Use fixed random seed", 
            value=st.session_state.seed_enabled, 
            key="seed_enabled"
        )
        random_seed = st.number_input(
            "Random seed", 
            min_value=0, max_value=999999, 
            value=int(st.session_state.random_seed), step=1,
            key="random_seed"
        )
    
    # DAMPING & CONTROL DEVICES
    with st.expander("Control Systems", expanded=True):
        damping = st.slider(
            "Inherent Damping (%)", 0.5, 15.0, 
            st.session_state.damping_pct, 
            key="damping_pct"
        ) / 100
        
        st.markdown("**Base Isolation**")
        base_iso = st.checkbox(
            "Enable Base Isolation", 
            value=st.session_state.base_iso, 
            key="base_iso"
        )
        if base_iso:
            col7, col8 = st.columns(2)
            with col7:
                iso_period = st.slider(
                    "Isolation Period (s)", 1.5, 4.0, 
                    st.session_state.iso_period, 0.1, 
                    key="iso_period"
                )
            with col8:
                iso_damping = st.slider(
                    "Isolation Damping", 0.05, 0.30, 
                    st.session_state.iso_damping, 0.01, 
                    key="iso_damping"
                )
        else:
            iso_period = st.session_state.iso_period
            iso_damping = st.session_state.iso_damping
        
        st.markdown("**Supplemental Damping**")
        damper_names = [d.value for d in DamperType]
        damper_selected = st.selectbox(
            "Damper Type",
            damper_names,
            index=damper_names.index(st.session_state.damper_selected),
            key="damper_selected",
        )
        damper_type = [d for d in DamperType if d.value == damper_selected][0]
        
        if damper_type == DamperType.TMD:
            col9, col10 = st.columns(2)
            with col9:
                tmd_mass_ratio = st.slider(
                    "TMD Mass Ratio", 0.01, 0.05, 
                    st.session_state.tmd_mass_ratio, 0.005, 
                    key="tmd_mass_ratio"
                )
            with col10:
                tmd_floor_default = min(int(st.session_state.tmd_floor), int(n_floors))
                tmd_floor = st.number_input(
                    "TMD Floor", 1, n_floors, 
                    tmd_floor_default, 
                    key="tmd_floor"
                )
        else:
            tmd_mass_ratio = st.session_state.tmd_mass_ratio
            tmd_floor = -1
        
        if damper_type == DamperType.VISCOUS:
            viscous_c = st.slider(
                "Damper Coefficient (kN·s/m)", 100, 5000, 
                st.session_state.viscous_c_kns, 
                key="viscous_c_kns"
            ) * 1000
        else:
            viscous_c = 1e5
    
    # ANALYSIS OPTIONS
    with st.expander("Analysis Options"):
        analysis_names = [a.value for a in AnalysisType]
        analysis_selected = st.selectbox(
            "Analysis Type",
            analysis_names,
            index=analysis_names.index(st.session_state.analysis_selected),
            key="analysis_selected",
        )
        analysis_type = [a for a in AnalysisType if a.value == analysis_selected][0]
        
        # Show warning for nonlinear with bidirectional
        if "Nonlinear" in analysis_selected:
            st.warning("Nonlinear analysis uses X-direction only. Bidirectional disabled.", icon="⚠️")
        
        if "Nonlinear" in analysis_selected:
            col11, col12 = st.columns(2)
            with col11:
                yield_drift = st.slider(
                    "Yield Drift (%)", 0.5, 2.0, 
                    st.session_state.yield_drift_pct, 
                    key="yield_drift_pct"
                ) / 100
            with col12:
                hardening = st.slider(
                    "Hardening Ratio", 0.01, 0.10, 
                    st.session_state.hardening, 0.01, 
                    key="hardening"
                )
        else:
            yield_drift = st.session_state.yield_drift_pct / 100
            hardening = st.session_state.hardening
        
        dt = st.select_slider(
            "Time Step (s)", 
            [0.005, 0.01, 0.02, 0.05], 
            value=st.session_state.dt, 
            key="dt"
        )
        
        st.markdown("**Output Options**")
        compute_spectra = st.checkbox(
            "Compute Response Spectra", 
            value=st.session_state.compute_spectra, 
            key="compute_spectra"
        )
        compute_fragility = st.checkbox(
            "Compute Fragility Curves", 
            value=st.session_state.compute_fragility, 
            key="compute_fragility"
        )
        compare_mode = st.checkbox(
            "Compare With/Without Control", 
            value=st.session_state.compare_mode, 
            key="compare_mode"
        )
    
    # IS 1893 CODE
    with st.expander("IS 1893 Design Parameters"):
        is_code_enabled = st.checkbox(
            "Enable IS 1893 spectrum", 
            value=st.session_state.is_code_enabled, 
            key="is_code_enabled"
        )
        is_zone = st.selectbox(
            "Seismic Zone",
            ["II", "III", "IV", "V"],
            index=["II", "III", "IV", "V"].index(st.session_state.is_zone),
            key="is_zone",
        )
        is_soil = st.selectbox(
            "Site Soil",
            ["Rock", "Medium", "Soft"],
            index=["Rock", "Medium", "Soft"].index(st.session_state.is_soil),
            key="is_soil",
        )
        is_importance = st.slider(
            "Importance Factor (I)", 1.0, 1.5, 
            float(st.session_state.is_importance), 0.05, 
            key="is_importance"
        )
        is_response_reduction = st.slider(
            "Response Reduction (R)", 3.0, 5.0, 
            float(st.session_state.is_response_reduction), 0.5, 
            key="is_response_reduction"
        )
        show_is_overlay = st.checkbox(
            "Show IS spectrum overlay", 
            value=st.session_state.show_is_overlay, 
            key="show_is_overlay"
        )
    
    st.divider()
    
    # RUN BUTTONS
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        quick_btn = st.button("⚡ Quick Run", use_container_width=True)
    with col_btn2:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    
    st.caption("Quick Run skips spectra, fragility, and comparison.")
    
    st.markdown("---")
    if 'simulation_results' in st.session_state and st.session_state.simulation_results is not None:
        st.success("✅ Results available")
    else:
        st.info("Ready to analyze")


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'run_count' not in st.session_state:
    st.session_state.run_count = 0

if reset_btn:
    st.session_state.simulation_results = None
    st.session_state.comparison_results = None
    st.rerun()


# =============================================================================
# MAIN EXECUTION ENGINE
# =============================================================================

if run_btn or quick_btn:
    st.session_state.run_count += 1
    run_id = f"run_{st.session_state.run_count}_{datetime.now().strftime('%H%M%S')}"
    
    quick_mode = bool(quick_btn) and not bool(run_btn)
    compute_spectra_run = compute_spectra and not quick_mode
    compute_fragility_run = compute_fragility and not quick_mode
    compare_mode_run = compare_mode and not quick_mode
    
    # For nonlinear analysis, force X-direction
    if "Nonlinear" in analysis_selected:
        direction = "X"
    
    progress_bar = st.progress(0, text="Initializing...")
    status_container = st.empty()
    
    logger.info(f"Starting analysis {run_id}: {analysis_selected}, {n_floors} floors, PGA={pga}g")
    
    try:
        with PerformanceLogger(f"analysis_{run_id}", logger), st.spinner("Analysis in progress..."):
            
            # --- A. CREATE PROPERTY OBJECTS ---
            progress_bar.progress(5, text="Building structural model...")
            
            # Compute scaled stiffness based on number of floors
            k_nominal = compute_scaled_stiffness(n_floors, mass, stiffness_factor)
            
            props = StructuralProperties(
                n_floors=n_floors,
                floor_mass=mass,
                floor_height=floor_height,
                building_width=width,
                building_depth=depth,
                column_stiffness=k_nominal,
                damping_ratio=damping,
                eccentricity=ecc / 100,
                yield_drift=yield_drift,
                hardening_ratio=hardening
            )
            
            controls = ControlDevices(
                base_isolated=base_iso,
            isolation_period=iso_period,
            isolation_damping=iso_damping,
            damper_type=damper_type,
            tmd_mass_ratio=tmd_mass_ratio,
            tmd_floor=tmd_floor - 1 if tmd_floor > 0 else -1,
            viscous_c=viscous_c
        )
        
        # --- B. GENERATE GROUND MOTION ---
        progress_bar.progress(15, text="Generating earthquake motion...")
        
        seed = int(random_seed) if seed_enabled else None
        
        dir_clean = direction.replace(" (Bidirectional)", "")
        if "XY" in direction:
            t, ag_x, ag_y = generate_bidirectional_motion(
                event_type, pga, duration, dt, soil, seed
            )
            ag_x = ag_x * scale_factor
            ag_y = ag_y * scale_factor
            ag = ag_x
        else:
            t, ag = generate_ground_motion(
                event_type, pga, duration, dt, soil, scale_factor, seed
            )
            ag_y = None
        
        # --- C. BUILD MATRICES ---
        progress_bar.progress(25, text="Assembling structural matrices...")
        
        M, K, _ = build_structural_matrices(props, controls)
        
        # Modal analysis
        periods, eig_vecs = perform_modal_analysis(M, K)
        
        # Check for unrealistic period
        period_warning = check_period_warning(periods[0], n_floors)
        
        # Rayleigh damping
        C = compute_rayleigh_damping(M, K, damping, controls)
        
        # --- D. RUN SOLVER ---
        progress_bar.progress(40, text="Solving equations of motion...")
        
        if analysis_type == AnalysisType.LINEAR:
            ag_input: Union[np.ndarray, Dict[str, np.ndarray]]
            if "XY" in direction:
                ag_input = {"x": ag_x, "y": ag_y}
            else:
                ag_input = ag
            results = newmark_linear_solver(M, K, C, ag_input, t, dir_clean)
            u = results["displacement"]
            v = results["velocity"]
            a = results["relative_acceleration"]
            a_abs = results["absolute_acceleration"]
            max_ductility = np.ones(n_floors)
            force_history = None
            disp_history = None
            residual_drift = np.zeros(n_floors)
        else:
            # Nonlinear solver (X-direction only)
            results = newmark_nonlinear_solver(
                M, K, C, ag, t, yield_drift, hardening, floor_height
            )
            u = results["displacement"]
            v = results["velocity"]
            a = results["relative_acceleration"]
            a_abs = results["absolute_acceleration"]
            max_ductility = results["max_ductility"]
            force_history = results["force_history"]
            disp_history = results["disp_history"]
            residual_drift = results["residual_drift"]
        
        # --- E. POST-PROCESSING ---
        progress_bar.progress(60, text="Computing response quantities...")
        
        # Roof response
        roof_idx = 3 * (n_floors - 1)
        roof_disp = u[:, roof_idx] * 100  # cm
        roof_vel = v[:, roof_idx] * 100  # cm/s
        roof_acc = a_abs[:, roof_idx] / 9.81  # g
        
        # Interstorey drifts
        drift_ratios, max_drifts_per_floor, max_drift_ratio = compute_interstorey_drifts(
            u, floor_height, n_floors
        )
        
        # Base shear
        base_shear = compute_base_shear(M, a_abs)
        max_base_shear = np.max(np.abs(base_shear))
        
        # Energy balance
        progress_bar.progress(70, text="Computing energy balance...")
        energy = calculate_energy_balance(M, K, C, u, v, ag, dt)
        
        # Response spectra
        if compute_spectra_run:
            progress_bar.progress(80, text="Computing response spectra...")
            ground_spectra = compute_response_spectrum(ag, dt)
            floor_spectra = compute_floor_response_spectrum(a_abs[:, roof_idx], dt)
        else:
            ground_spectra = {"periods": np.array([1]), "Sa": np.array([0]), "Sd": np.array([0]), "Sv": np.array([0])}
            floor_spectra = ground_spectra
        
        # Fragility curves
        if compute_fragility_run:
            progress_bar.progress(85, text="Computing fragility curves...")
            fragility = compute_fragility_curves(np.array([max_drift_ratio]), np.array([pga]))
        else:
            fragility = {}
        
        # Performance assessment
        perf_level, perf_desc, perf_color = assess_performance_level(max_drift_ratio)
        
        # Damage index
        energy_ratio = energy["hysteretic"][-1] / max(energy["input"][-1], 1)
        damage_idx = compute_damage_index(max_drift_ratio, max_ductility, energy_ratio)
        
        # Residual drift assessment
        residual_assessment, residual_color = compute_residual_drift_assessment(residual_drift)
        
        # IS 1893 calculations
        total_weight = n_floors * mass * 9.81
        if is_code_enabled:
            Ah, Vb = compute_design_base_shear(
                total_weight, periods[0], is_soil, is_zone,
                is_importance, is_response_reduction
            )
        else:
            Ah, Vb = 0.0, 0.0
        
        is_drift_limit_pct = get_drift_limit_is1893()
        
        # Recommendations
        has_control = base_iso or damper_type != DamperType.NONE
        recommendations = generate_recommendations(
            {"max_drift_ratio": max_drift_ratio, "periods": periods, "residual_drift": residual_drift},
            n_floors, has_control
        )
        
        # Add period warning to recommendations if applicable
        if period_warning:
            recommendations.insert(0, period_warning)
        
        # --- F. COMPARISON MODE ---
        if compare_mode_run:
            progress_bar.progress(90, text="Running comparison analysis...")
            
            controls_none = ControlDevices(base_isolated=False, damper_type=DamperType.NONE)
            M_none, K_none, _ = build_structural_matrices(props, controls_none)
            C_none = compute_rayleigh_damping(M_none, K_none, damping, controls_none)
            
            results_none = newmark_linear_solver(M_none, K_none, C_none, ag, t, dir_clean)
            u_none = results_none["displacement"]
            
            _, _, max_drift_none = compute_interstorey_drifts(u_none, floor_height, n_floors)
            roof_disp_none = u_none[:, roof_idx] * 100
            
            comparison_data = {
                "max_drift_controlled": max_drift_ratio,
                "max_drift_uncontrolled": max_drift_none,
                "roof_disp_controlled": roof_disp,
                "roof_disp_uncontrolled": roof_disp_none,
                "reduction_percent": (1 - max_drift_ratio / max(max_drift_none, 0.01)) * 100
            }
        else:
            comparison_data = {}
        
        progress_bar.progress(95, text="Finalizing results...")
        
        # --- G. SAVE RESULTS ---
        st.session_state.simulation_results = {
            't': t,
            'ag': ag,
            'ag_y': ag_y,
            'roof_disp': roof_disp,
            'roof_vel': roof_vel,
            'roof_acc': roof_acc,
            'u': u,
            'v': v,
            'a_abs': a_abs,
            'base_shear': base_shear,
            'drifts': None,  # Not storing full array
            'drift_ratios': drift_ratios,
            'max_drift_ratio': max_drift_ratio,
            'max_drifts_per_floor': max_drifts_per_floor,
            'periods': periods,
            'eig_vecs': eig_vecs,
            'freqs': 1.0 / periods[:min(10, len(periods))],
            'max_base_shear': max_base_shear,
            'energy': energy,
            'ground_spectra': ground_spectra,
            'floor_spectra': floor_spectra,
            'fragility': fragility,
            'perf_level': perf_level,
            'perf_desc': perf_desc,
            'perf_color': perf_color,
            'damage_index': damage_idx,
            'max_ductility': max_ductility,
            'force_history': force_history,
            'disp_history': disp_history,
            'residual_drift': residual_drift,
            'residual_assessment': residual_assessment,
            'residual_color': residual_color,
            'recommendations': recommendations,
            'comparison': comparison_data,
            'is_code': {
                'enabled': is_code_enabled,
                'zone': is_zone,
                'soil': is_soil,
                'importance': is_importance,
                'response_reduction': is_response_reduction,
                'Ah': Ah,
                'Vb': Vb,
                'drift_limit_pct': is_drift_limit_pct,
            },
            'n_floors': n_floors,
            'floor_height': floor_height,
            'width': width,
            'analysis_type': analysis_type.value,
            'event_type': event_type.value,
            'pga': pga,
            'controls': {
                'base_isolated': base_iso,
                'damper_type': damper_type.value
            },
            'timestamp': datetime.now().isoformat(),
            'run_id': run_id
        }
        
        progress_bar.progress(100, text="Analysis complete")
        status_container.success(f"✅ Analysis complete. Run #{st.session_state.run_count}")
        logger.info(f"Analysis {run_id} completed successfully")
    
    except ValidationError as e:
        progress_bar.empty()
        logger.error(f"Validation error in analysis {run_id}: {e}")
        st.error(f"⚠️ Input validation error: {e}")
        st.info("Please check your input parameters and try again.")
    
    except SolverError as e:
        progress_bar.empty()
        logger.error(f"Solver error in analysis {run_id}: {e}")
        st.error(f"❌ Solver error: {e}")
        st.warning("Try reducing the time step or adjusting structural parameters.")
    
    except SeismicAppError as e:
        progress_bar.empty()
        logger.error(f"Application error in analysis {run_id}: {e}")
        st.error(f"❌ Analysis error: {e}")
    
    except Exception as e:
        progress_bar.empty()
        logger.exception(f"Unexpected error in analysis {run_id}")
        st.error(f"❌ Unexpected error: {type(e).__name__}: {e}")
        if app_config.debug_mode:
            st.exception(e)


# =============================================================================
# DASHBOARD UI
# =============================================================================

if st.session_state.simulation_results is not None:
    res = st.session_state.simulation_results
    t = res['t']
    ag = res['ag']
    n_floors = res['n_floors']
    floor_height = res['floor_height']
    
    # ========== TOP KPI ROW ==========
    st.markdown("### Key Performance Indicators")
    
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    
    with c1:
        drift_delta = "Danger" if res['max_drift_ratio'] > 2.0 else (
            "Warning" if res['max_drift_ratio'] > 1.0 else "Safe"
        )
        st.metric("Max Drift Ratio", f"{res['max_drift_ratio']:.2f}%", drift_delta)
    
    with c2:
        st.metric("Peak Roof Accel", f"{np.max(np.abs(res['roof_acc'])):.3f} g")
    
    with c3:
        st.metric("Max Base Shear", f"{res['max_base_shear']:.0f} kN")
    
    with c4:
        st.metric("Fund. Period T₁", f"{res['periods'][0]:.3f} s")
    
    with c5:
        st.metric("Damage Index", f"{res['damage_index']:.2f}")
    
    with c6:
        st.markdown(f"""
        <div style="background: {res['perf_color']}; color: white; padding: 10px 15px; 
             border-radius: 10px; text-align: center; font-weight: 700;">
            {res['perf_level']}
        </div>
        """, unsafe_allow_html=True)
    
    with c7:
        if res.get('is_code', {}).get('enabled'):
            st.metric("IS Base Shear", f"{res['is_code']['Vb'] / 1000:.1f} kN")
        else:
            st.metric("IS Base Shear", "N/A")
    
    st.markdown("---")
    
    # ========== MAIN TABS ==========
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Dashboard",
        "🎬 Animation",
        "⚡ Advanced Physics",
        "📈 Response Spectra",
        "⚖️ Comparison",
        "🎯 Performance",
        "📥 Export"
    ])
    
    # Import tabs from separate module
    from ui_tabs import (
        render_dashboard_tab, render_animation_tab, render_physics_tab,
        render_spectra_tab, render_comparison_tab, render_performance_tab,
        render_export_tab
    )
    
    with tab1:
        render_dashboard_tab(res, t, ag, n_floors, floor_height)
    
    with tab2:
        render_animation_tab(res, t, n_floors, floor_height)
    
    with tab3:
        render_physics_tab(res, t)
    
    with tab4:
        render_spectra_tab(res, show_is_overlay)
    
    with tab5:
        render_comparison_tab(res, t)
    
    with tab6:
        render_performance_tab(res)
    
    with tab7:
        render_export_tab(res, t, ag, n_floors)

else:
    # ========== WELCOME SCREEN ==========
    st.markdown("""
    <div style="text-align: center; padding: 30px;">
        <h2 style="color: #e2e8f0;">Welcome to the Seismic Analysis Tool</h2>
        <p style="color: #94a3b8;">Configure your building and earthquake parameters in the sidebar, then click <b>Run Analysis</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature cards
    cols = st.columns(4)
    for i, (title, desc, color) in enumerate(WELCOME_FEATURES):
        with cols[i]:
            st.markdown(get_feature_card(title, desc, color), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Educational disclaimer
    st.info("""
    **📚 Educational Purpose**
    
    This tool is designed for Civil Engineering education to help students understand:
    - Structural dynamics and modal analysis
    - Earthquake response of multi-story buildings
    - Effects of control devices (base isolation, dampers)
    - Performance-based seismic design concepts
    
    **⚠️ All ground motions are synthetically generated** and should not be used for actual design.
    """)
