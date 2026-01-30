"""
Seismic Analysis Dashboard

An educational tool for MDOF shear building seismic analysis.
Refactored into modular components for maintainability.

Modules:
    - models: Data classes and enumerations
    - config: Configuration and defaults
    - styles: CSS styling and UI templates
    - ground_motion: Earthquake generation
    - structural: Matrix assembly and modal analysis
    - solvers: Time-history integration (Newmark-Beta)
    - postprocess: Response processing and assessment
    - design_codes: IS 1893 calculations
    - plotting: Visualization functions
    - report: PDF report generation
    - ui_tabs: Dashboard tab renderers
    - main: Streamlit application entry point

Usage:
    streamlit run main.py

Author: Educational Seismic Analysis Tool
Version: 2.0.0 (Modular Refactor)
"""

__version__ = "2.0.0"
__author__ = "Educational Seismic Analysis Tool"

# Key fixes in this version:
# 1. SeismicEvent names now clearly indicate synthetic data (academic honesty)
# 2. Nonlinear analysis locked to X-direction only (physics integrity)
# 3. Response spectrum computation vectorized (algorithmic optimization)
# 4. Plot decimation prevents browser crashes (browser stability)
# 5. Stiffness scaling based on number of floors (realistic defaults)
# 6. Residual drift metric added (missing feature)
