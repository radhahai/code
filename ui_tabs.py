"""
ui_tabs.py - Dashboard Tab Renderers

This module contains the rendering functions for each dashboard tab.
Separated for code organization and maintainability.
"""

import json
from datetime import datetime
from typing import Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq

from plotting import (
    create_time_history_plot, create_base_shear_plot,
    create_drift_profile_plot, create_drift_heatmap,
    create_energy_plot, create_hysteresis_plot, create_phase_plane_plot,
    create_response_spectrum_plot, create_3d_mode_shape_plot,
    decimate_data
)
from design_codes import compute_is1893_spectrum
from report import generate_pdf_report


# =============================================================================
# TAB 1: DASHBOARD
# =============================================================================

def render_dashboard_tab(
    res: Dict[str, Any],
    t: np.ndarray,
    ag: np.ndarray,
    n_floors: int,
    floor_height: float
) -> None:
    """Render the main dashboard tab."""
    
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        # Time history plots with decimation
        fig1 = create_time_history_plot(
            t, res['roof_disp'], res['roof_acc'], ag
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_side:
        # Drift profile
        st.markdown("#### Interstorey Drift Profile")
        
        is_drift_limit = res.get('is_code', {}).get('drift_limit_pct') if res.get('is_code', {}).get('enabled') else None
        fig_drift = create_drift_profile_plot(res['max_drifts_per_floor'], is_drift_limit)
        st.plotly_chart(fig_drift, use_container_width=True)
        
        # Base shear history
        st.markdown("#### Base Shear History")
        fig_shear = create_base_shear_plot(t, res['base_shear'])
        st.plotly_chart(fig_shear, use_container_width=True)
        
        # Drift heatmap
        st.markdown("#### Drift Ratio Heatmap")
        floor_labels = [f"F{i+1}" for i in range(len(res['max_drifts_per_floor']))]
        fig_heat = create_drift_heatmap(t, res['drift_ratios'])
        st.plotly_chart(fig_heat, use_container_width=True)


# =============================================================================
# TAB 2: ANIMATION
# =============================================================================

def render_animation_tab(
    res: Dict[str, Any],
    t: np.ndarray,
    n_floors: int,
    floor_height: float
) -> None:
    """Render the animation tab with building response and mode shapes."""
    
    col_anim, col_mode = st.columns([1, 1])
    
    with col_anim:
        st.markdown("#### Building Response Animation")
        
        # Create animation frames (limited for performance)
        step = max(1, len(t) // 100)
        roof_disp = res['roof_disp']
        u = res.get('u', np.zeros((len(t), n_floors * 3)))
        u_x = u[:, 0::3] if u.shape[1] >= n_floors else np.zeros((len(t), n_floors))
        # Limit to building floors only (exclude TMD DOFs if present)
        if u_x.shape[1] > n_floors:
            u_x = u_x[:, :n_floors]
        
        max_disp = max(np.max(np.abs(roof_disp)), 1)
        heights = np.arange(0, (n_floors + 1) * floor_height, floor_height)
        
        frames = []
        for k in range(0, len(t), step):
            floor_disps = np.insert(u_x[k] * 100, 0, 0) if u_x.shape[1] == n_floors else np.zeros(n_floors + 1)
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=floor_disps, y=heights, 
                    mode='lines+markers',
                    line=dict(width=4, color='#00CC96'),
                    marker=dict(size=10)
                )],
                name=str(k)
            ))
        
        fig_anim = go.Figure(
            data=[go.Scatter(
                x=np.zeros(n_floors + 1), y=heights,
                mode='lines+markers',
                line=dict(width=4, color='#00CC96'),
                marker=dict(size=10)
            )],
            layout=go.Layout(
                xaxis=dict(range=[-max_disp * 1.5, max_disp * 1.5], title="Displacement (cm)"),
                yaxis=dict(range=[0, n_floors * floor_height * 1.1], title="Height (m)"),
                template="plotly_dark",
                height=500,
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    y=1.15,
                    x=0.5,
                    xanchor="center",
                    buttons=[
                        dict(label="▶ Play", method="animate",
                             args=[None, {"frame": {"duration": 50, "redraw": True},
                                          "fromcurrent": True}]),
                        dict(label="⏸ Pause", method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate"}])
                    ]
                )]
            ),
            frames=frames
        )
        
        # Add slider
        fig_anim.update_layout(
            sliders=[dict(
                active=0,
                steps=[dict(
                    args=[[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    method="animate",
                    label=f"{int(k * (t[1]-t[0]))}" if k < len(t) else ""
                ) for k, f in enumerate(frames)],
                currentvalue=dict(prefix="Time: ", suffix=" s"),
                len=0.9
            )]
        )
        
        st.plotly_chart(fig_anim, use_container_width=True)
    
    with col_mode:
        st.markdown("#### 3D Mode Shape Visualization")
        
        mode_options = [
            f"Mode {i+1} (T={res['periods'][i]:.3f}s, f={1/res['periods'][i]:.2f}Hz)"
            for i in range(min(6, len(res['periods'])))
        ]
        mode_select = st.selectbox("Select Mode Shape", mode_options)
        mode_idx = int(mode_select.split(" ")[1]) - 1
        
        # Extract mode shape
        vec = res['eig_vecs'][:, mode_idx]
        fig_3d = create_3d_mode_shape_plot(vec, floor_height, res['width'], n_floors)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Modal information
        st.markdown("#### Modal Information")
        modal_df = pd.DataFrame({
            "Mode": [f"Mode {i+1}" for i in range(min(5, len(res['periods'])))],
            "Period (s)": [f"{res['periods'][i]:.4f}" for i in range(min(5, len(res['periods'])))],
            "Frequency (Hz)": [f"{1/res['periods'][i]:.3f}" for i in range(min(5, len(res['periods'])))]
        })
        st.dataframe(modal_df, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 3: ADVANCED PHYSICS
# =============================================================================

def render_physics_tab(res: Dict[str, Any], t: np.ndarray) -> None:
    """Render the advanced physics tab."""
    
    st.markdown("### Advanced Physics Analysis")
    
    col_energy, col_hysteresis = st.columns([1, 1])
    
    with col_energy:
        st.markdown("#### Energy Balance")
        
        fig_eng = create_energy_plot(t, res['energy'])
        st.plotly_chart(fig_eng, use_container_width=True)
        
        # Energy summary
        st.markdown("##### Energy Summary at End of Motion")
        energy = res['energy']
        energy_summary = {
            "Component": ["Input", "Kinetic", "Strain", "Damping", "Hysteretic"],
            "Value (kJ)": [
                f"{energy['input'][-1]/1000:.2f}",
                f"{energy['kinetic'][-1]/1000:.2f}",
                f"{energy['strain'][-1]/1000:.2f}",
                f"{energy['damping'][-1]/1000:.2f}",
                f"{energy['hysteretic'][-1]/1000:.2f}"
            ],
            "Percentage": [
                f"{100:.1f}%",
                f"{energy['kinetic'][-1]/max(energy['input'][-1],1)*100:.1f}%",
                f"{energy['strain'][-1]/max(energy['input'][-1],1)*100:.1f}%",
                f"{energy['damping'][-1]/max(energy['input'][-1],1)*100:.1f}%",
                f"{energy['hysteretic'][-1]/max(energy['input'][-1],1)*100:.1f}%"
            ]
        }
        st.dataframe(pd.DataFrame(energy_summary), use_container_width=True, hide_index=True)
    
    with col_hysteresis:
        st.markdown("#### Force-Deformation Response")
        
        if res['force_history'] is not None and res['disp_history'] is not None:
            # Nonlinear hysteresis
            floor_select = st.selectbox(
                "Select Floor",
                [f"Floor {i+1}" for i in range(len(res['max_ductility']))],
                key="hyst_floor"
            )
            floor_idx = int(floor_select.split(" ")[1]) - 1
            
            fig_hyst = create_hysteresis_plot(
                res['disp_history'], res['force_history'], floor_idx
            )
            st.plotly_chart(fig_hyst, use_container_width=True)
            
            # Ductility demands
            st.markdown("##### Ductility Demands")
            ductility_df = pd.DataFrame({
                "Floor": [f"F{i+1}" for i in range(len(res['max_ductility']))],
                "Max Ductility": [f"{d:.2f}" for d in res['max_ductility']]
            })
            st.dataframe(ductility_df, use_container_width=True, hide_index=True)
        else:
            # Linear analysis - show phase plane
            st.markdown("*Linear analysis - showing phase plane*")
            fig_phase = create_phase_plane_plot(res['roof_disp'], res['roof_vel'])
            st.plotly_chart(fig_phase, use_container_width=True)
        
        # Fourier spectrum
        st.markdown("#### Fourier Amplitude Spectrum")
        
        roof_acc = res['roof_acc'] * 9.81
        dt = t[1] - t[0]
        n_fft = len(roof_acc)
        
        fft_vals = np.abs(fft(roof_acc))[:n_fft//2]
        freqs_fft = fftfreq(n_fft, dt)[:n_fft//2]
        
        # Decimate for plotting
        mask = freqs_fft < 15
        freqs_plot = freqs_fft[mask]
        fft_plot = fft_vals[mask]
        
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(
            x=freqs_plot,
            y=fft_plot,
            fill='tozeroy',
            line=dict(color='#06b6d4', width=1),
            fillcolor='rgba(6, 182, 212, 0.3)'
        ))
        
        # Mark natural frequencies
        for i, period in enumerate(res['periods'][:3]):
            freq = 1 / period
            if freq < 15:
                fig_fft.add_vline(x=freq, line_dash="dash", line_color="#f59e0b",
                                 annotation_text=f"f{i+1}")
        
        fig_fft.update_layout(
            template="plotly_dark",
            height=250,
            xaxis_title="Frequency (Hz)",
            yaxis_title="Amplitude",
            showlegend=False
        )
        
        st.plotly_chart(fig_fft, use_container_width=True)


# =============================================================================
# TAB 4: RESPONSE SPECTRA
# =============================================================================

def render_spectra_tab(res: Dict[str, Any], show_is_overlay: bool) -> None:
    """Render the response spectra tab."""
    
    st.markdown("### Response Spectra Analysis")
    
    col_spec1, col_spec2 = st.columns(2)
    
    with col_spec1:
        st.markdown("#### Ground Motion Response Spectrum")
        
        ground_spec = res['ground_spectra']
        
        # IS 1893 spectrum overlay
        is_spec = None
        if res.get('is_code', {}).get('enabled') and show_is_overlay:
            is_spec = compute_is1893_spectrum(ground_spec['periods'], res['is_code']['soil'])
        
        fig_spec_g = create_response_spectrum_plot(
            ground_spec, res['periods'][:3], is_spec
        )
        st.plotly_chart(fig_spec_g, use_container_width=True)
    
    with col_spec2:
        st.markdown("#### Floor Response Spectrum (Top Floor)")
        
        floor_spec = res['floor_spectra']
        
        fig_spec_f = go.Figure()
        fig_spec_f.add_trace(go.Scatter(
            x=floor_spec['periods'],
            y=floor_spec['Sa'],
            line=dict(color='#8b5cf6', width=2),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.2)',
            name='Floor Sa'
        ))
        
        fig_spec_f.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Period (s)",
            yaxis_title="Sa (g)",
            xaxis_type="log",
            title="Equipment Design Spectrum"
        )
        
        st.plotly_chart(fig_spec_f, use_container_width=True)
        
        st.info("Use Floor Response Spectra for designing equipment and secondary systems mounted on the structure.")
    
    # Spectral values table
    st.markdown("#### Spectral Values at Structure Natural Periods")
    
    ground_spec = res['ground_spectra']
    spectral_table = []
    for i, T in enumerate(res['periods'][:5]):
        idx = np.argmin(np.abs(ground_spec['periods'] - T))
        spectral_table.append({
            "Mode": f"Mode {i+1}",
            "Period T (s)": f"{T:.4f}",
            "Sa (g)": f"{ground_spec['Sa'][idx]:.4f}",
            "Sd (cm)": f"{ground_spec['Sd'][idx]*100:.3f}"
        })
    
    st.dataframe(pd.DataFrame(spectral_table), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 5: COMPARISON
# =============================================================================

def render_comparison_tab(res: Dict[str, Any], t: np.ndarray) -> None:
    """Render the comparison tab."""
    
    st.markdown("### Control System Comparison")
    
    if res['comparison'] and len(res['comparison']) > 0:
        comp = res['comparison']
        
        # Comparison metrics
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        
        with col_comp1:
            st.metric(
                "Drift Reduction",
                f"{comp['reduction_percent']:.1f}%",
                delta=f"-{comp['max_drift_uncontrolled'] - comp['max_drift_controlled']:.2f}%",
                delta_color="inverse"
            )
        
        with col_comp2:
            st.metric("With Control", f"{comp['max_drift_controlled']:.2f}%")
        
        with col_comp3:
            st.metric("Without Control", f"{comp['max_drift_uncontrolled']:.2f}%")
        
        # Time history comparison with decimation
        t_d, disp_uncontrolled = decimate_data(t, comp['roof_disp_uncontrolled'])
        _, disp_controlled = decimate_data(t, comp['roof_disp_controlled'])
        
        fig_comp = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Roof Displacement Comparison", "Effectiveness")
        )
        
        fig_comp.add_trace(go.Scatter(
            x=t_d, y=disp_uncontrolled,
            name="Without Control",
            line=dict(color='#ef4444', width=1.5)
        ), row=1, col=1)
        
        fig_comp.add_trace(go.Scatter(
            x=t_d, y=disp_controlled,
            name="With Control",
            line=dict(color='#22c55e', width=1.5)
        ), row=1, col=1)
        
        # Reduction over time
        reduction = (np.abs(disp_uncontrolled) - np.abs(disp_controlled)) / \
                   (np.abs(disp_uncontrolled) + 0.01) * 100
        
        fig_comp.add_trace(go.Scatter(
            x=t_d, y=reduction,
            name="Reduction %",
            fill='tozeroy',
            line=dict(color='#3b82f6', width=1),
            fillcolor='rgba(59, 130, 246, 0.3)'
        ), row=2, col=1)
        
        fig_comp.update_layout(
            template="plotly_dark",
            height=500,
            hovermode="x unified"
        )
        fig_comp.update_yaxes(title_text="Displacement (cm)", row=1, col=1)
        fig_comp.update_yaxes(title_text="Reduction (%)", row=2, col=1)
        fig_comp.update_xaxes(title_text="Time (s)", row=2, col=1)
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.success(f"Control system provides **{comp['reduction_percent']:.1f}%** reduction in maximum drift.")
    else:
        st.info("Enable 'Compare With/Without Control' in the sidebar and run analysis to see comparison.")
        
        st.markdown("""
        #### What comparison analysis shows:
        - **Drift Reduction**: How much the control system reduces interstorey drift
        - **Time History Comparison**: Side-by-side view of controlled vs uncontrolled response
        - **Effectiveness Over Time**: How well the control system performs throughout the earthquake
        """)


# =============================================================================
# TAB 6: PERFORMANCE
# =============================================================================

def render_performance_tab(res: Dict[str, Any]) -> None:
    """Render the performance assessment tab."""
    
    st.markdown("### Performance-Based Assessment")
    
    col_perf1, col_perf2 = st.columns([1, 1])
    
    with col_perf1:
        st.markdown("#### Performance Level")
        
        # Gauge visualization
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=res['max_drift_ratio'],
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': 2.0, 'decreasing': {'color': "#22c55e"}},
            gauge={
                'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': res['perf_color']},
                'bgcolor': "rgba(30, 41, 59, 0.8)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.5], 'color': 'rgba(34, 197, 94, 0.3)'},
                    {'range': [0.5, 1.0], 'color': 'rgba(132, 204, 22, 0.3)'},
                    {'range': [1.0, 2.0], 'color': 'rgba(234, 179, 8, 0.3)'},
                    {'range': [2.0, 4.0], 'color': 'rgba(249, 115, 22, 0.3)'},
                    {'range': [4.0, 5.0], 'color': 'rgba(239, 68, 68, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': res['max_drift_ratio']
                }
            },
            title={'text': "Max Drift Ratio (%)"}
        ))
        
        fig_gauge.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Performance level description
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {res['perf_color']}33, {res['perf_color']}11);
                    border: 2px solid {res['perf_color']}; border-radius: 12px; padding: 20px; text-align: center;">
            <h3 style="color: {res['perf_color']}; margin: 0;">{res['perf_level']}</h3>
            <p style="color: #94a3b8; margin: 10px 0 0 0;">{res['perf_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Residual Drift (for nonlinear analysis)
        residual_drift = res.get('residual_drift', np.array([]))
        if len(residual_drift) > 0 and np.any(residual_drift != 0):
            st.markdown("#### Residual Drift (Permanent Deformation)")
            
            max_residual = np.max(np.abs(residual_drift))
            st.metric("Max Residual Drift", f"{max_residual:.3f}%")
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {res['residual_color']}33, {res['residual_color']}11);
                        border: 1px solid {res['residual_color']}; border-radius: 10px; padding: 15px;">
                <p style="color: {res['residual_color']}; margin: 0; font-weight: 600;">
                    {res['residual_assessment']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Residual drift per floor
            st.markdown("##### Residual Drift by Floor")
            residual_df = pd.DataFrame({
                "Floor": [f"F{i+1}" for i in range(len(residual_drift))],
                "Residual Drift (%)": [f"{rd:.4f}" for rd in residual_drift]
            })
            st.dataframe(residual_df, use_container_width=True, hide_index=True)
    
    with col_perf2:
        st.markdown("#### Damage Assessment")
        
        # Damage index visualization
        damage_level = (
            "Minor" if res['damage_index'] < 0.25 else
            "Moderate" if res['damage_index'] < 0.5 else
            "Severe" if res['damage_index'] < 0.75 else "Extreme"
        )
        
        fig_damage = go.Figure(go.Pie(
            values=[res['damage_index'], 1 - res['damage_index']],
            labels=['Damage', 'Remaining Capacity'],
            marker_colors=[res['perf_color'], '#1e293b'],
            hole=0.7,
            textinfo='none'
        ))
        
        fig_damage.add_annotation(
            text=f"<b>{res['damage_index']:.2f}</b><br>{damage_level}",
            x=0.5, y=0.5,
            font_size=20,
            showarrow=False,
            font_color='white'
        )
        
        fig_damage.update_layout(
            template="plotly_dark",
            height=250,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig_damage, use_container_width=True)
        
        # Fragility curves
        if res['fragility'] and 'fragility' in res['fragility']:
            st.markdown("#### Fragility Curves")
            
            fig_frag = go.Figure()
            colors = ['#22c55e', '#84cc16', '#eab308', '#ef4444']
            
            for i, (state, prob) in enumerate(res['fragility']['fragility'].items()):
                fig_frag.add_trace(go.Scatter(
                    x=res['fragility']['pga_range'],
                    y=prob,
                    name=state,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig_frag.add_vline(
                x=res['pga'], line_dash="dash", line_color="white",
                annotation_text=f"Design PGA = {res['pga']}g"
            )
            
            fig_frag.update_layout(
                template="plotly_dark",
                height=300,
                xaxis_title="PGA (g)",
                yaxis_title="P(Damage > State)",
                legend=dict(orientation="h", y=1.1)
            )
            
            st.plotly_chart(fig_frag, use_container_width=True)
    
    # Recommendations
    st.markdown("---")
    st.markdown("#### Recommendations")
    
    for rec in res['recommendations']:
        if "CRITICAL" in rec or "🚨" in rec:
            st.error(rec)
        elif "WARNING" in rec or "⚠️" in rec:
            st.warning(rec)
        elif "Suggestion" in rec or "💡" in rec:
            st.info(rec)
        elif "NOTE" in rec or "ℹ️" in rec:
            st.info(rec)
        elif "Excellent" in rec or "✅" in rec:
            st.success(rec)
        elif rec.strip().startswith("•"):
            st.markdown(rec)
        else:
            st.markdown(rec)
    
    # IS 1893 Summary
    if res.get('is_code', {}).get('enabled'):
        st.markdown("---")
        st.markdown("#### IS 1893 Summary")
        is_summary = pd.DataFrame({
            "Parameter": [
                "Zone", "Soil", "Importance (I)", "Response Reduction (R)",
                "Ah", "Design Base Shear (kN)", "Drift Limit (%)"
            ],
            "Value": [
                res['is_code']['zone'],
                res['is_code']['soil'],
                f"{res['is_code']['importance']:.2f}",
                f"{res['is_code']['response_reduction']:.2f}",
                f"{res['is_code']['Ah']:.4f}",
                f"{res['is_code']['Vb'] / 1000:.1f}",
                f"{res['is_code']['drift_limit_pct']:.2f}",
            ],
        })
        st.dataframe(is_summary, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 7: EXPORT
# =============================================================================

def render_export_tab(
    res: Dict[str, Any],
    t: np.ndarray,
    ag: np.ndarray,
    n_floors: int
) -> None:
    """Render the export tab."""
    
    st.markdown("### Export Results")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("#### Time History Data")
        
        # Create dataframe
        df_results = pd.DataFrame({
            "Time_s": t,
            "Ground_Acc_X_g": ag / 9.81,
            "Roof_Disp_cm": res['roof_disp'],
            "Roof_Vel_cm_s": res['roof_vel'],
            "Roof_Acc_g": res['roof_acc'],
            "Base_Shear_kN": res['base_shear']
        })
        if res.get('ag_y') is not None:
            df_results["Ground_Acc_Y_g"] = res['ag_y'] / 9.81
        
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Time History CSV",
            data=csv,
            file_name=f"seismic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.dataframe(df_results.head(20), use_container_width=True)
    
    with col_exp2:
        st.markdown("#### Analysis Summary")
        
        summary_params = [
            "Analysis Type", "Earthquake Event", "PGA (g)", "Duration (s)",
            "Number of Floors", "Fundamental Period (s)", "Max Drift Ratio (%)",
            "Max Base Shear (kN)", "Performance Level", "Damage Index",
            "Base Isolation", "Damper Type",
        ]
        summary_values = [
            res['analysis_type'], res['event_type'], f"{res['pga']:.2f}",
            f"{t[-1]:.1f}", str(n_floors), f"{res['periods'][0]:.4f}",
            f"{res['max_drift_ratio']:.3f}", f"{res['max_base_shear']:.1f}",
            res['perf_level'], f"{res['damage_index']:.3f}",
            "Yes" if res['controls']['base_isolated'] else "No",
            res['controls']['damper_type'],
        ]
        
        # Add residual drift for nonlinear
        residual_drift = res.get('residual_drift', np.array([]))
        if len(residual_drift) > 0 and np.any(residual_drift != 0):
            summary_params.append("Max Residual Drift (%)")
            summary_values.append(f"{np.max(np.abs(residual_drift)):.4f}")
        
        if res.get('is_code', {}).get('enabled'):
            summary_params.extend([
                "IS Zone", "IS Soil", "IS Importance (I)",
                "IS Response Reduction (R)", "IS Ah", "IS Base Shear (kN)",
            ])
            summary_values.extend([
                res['is_code']['zone'], res['is_code']['soil'],
                f"{res['is_code']['importance']:.2f}",
                f"{res['is_code']['response_reduction']:.2f}",
                f"{res['is_code']['Ah']:.4f}",
                f"{res['is_code']['Vb'] / 1000:.1f}",
            ])
        
        df_summary = pd.DataFrame({
            "Parameter": summary_params,
            "Value": summary_values,
        })
        
        summary_csv = df_summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Summary CSV",
            data=summary_csv,
            file_name=f"seismic_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
        
        # JSON export
        st.markdown("#### Full Results (JSON)")
        
        json_export = {
            "timestamp": res['timestamp'],
            "analysis_type": res['analysis_type'],
            "event_type": res['event_type'],
            "pga": res['pga'],
            "n_floors": res['n_floors'],
            "periods": res['periods'][:5].tolist(),
            "max_drift_ratio": res['max_drift_ratio'],
            "max_base_shear": res['max_base_shear'],
            "performance_level": res['perf_level'],
            "damage_index": res['damage_index'],
            "residual_drift_max": float(np.max(np.abs(res.get('residual_drift', [0])))),
            "controls": res['controls'],
            "recommendations": res['recommendations']
        }
        
        json_str = json.dumps(json_export, indent=2)
        st.download_button(
            label="📥 Download Full Results JSON",
            data=json_str,
            file_name=f"seismic_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    st.markdown("---")
    st.markdown("#### PDF Report")
    
    if st.button("📄 Generate PDF Report", use_container_width=True):
        with st.spinner("Generating PDF..."):
            pdf_bytes = generate_pdf_report(res)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"seismic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )
