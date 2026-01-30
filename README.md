# Seismic Analysis Tool

Advanced MDOF Shear Building Simulation for Structural Engineering Education.

## 🏗️ Overview

This application provides a comprehensive seismic analysis simulation for multi-degree-of-freedom (MDOF) shear buildings. It's designed for educational purposes, allowing users to explore:

- Linear and nonlinear time history analysis
- Base isolation and damper effects
- Response spectrum analysis
- Performance-based design metrics
- IS 1893 code compliance checks

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

## 📋 Requirements

- Python 3.10+
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Pandas >= 2.0.0
- Streamlit >= 1.28.0
- Plotly >= 5.15.0
- ReportLab >= 4.0.0

## 🏭 Production Deployment

### Environment Configuration

Set the `SEISMIC_APP_ENV` environment variable:

```bash
# Development (default)
export SEISMIC_APP_ENV=development

# Staging
export SEISMIC_APP_ENV=staging

# Production
export SEISMIC_APP_ENV=production
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SEISMIC_APP_ENV` | Environment mode | `development` |
| `LOG_LEVEL` | Logging level | `INFO` (prod), `DEBUG` (dev) |
| `MAX_WORKERS` | Parallel workers | `4` |
| `CACHE_TTL` | Cache time-to-live (s) | `3600` |

### Health Checks

Run the health check to verify the installation:

```bash
python health.py
```

This validates:
- All dependencies are installed
- Numerical backend is working correctly
- Core functionality (matrix assembly, ground motion, modal analysis)

### Logging

Logs are automatically configured based on environment:

- **Development**: Console output with colors, DEBUG level
- **Production**: JSON format with file rotation, INFO level

Log files are stored in `logs/` directory (production only).

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV SEISMIC_APP_ENV=production
EXPOSE 8501

HEALTHCHECK CMD python health.py || exit 1

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

## 📁 Project Structure

```
seismic_app/
├── main.py              # Main Streamlit application
├── config.py            # Configuration management
├── models.py            # Data models and enums
├── structural.py        # Structural matrix assembly
├── solvers.py           # Time integration solvers
├── ground_motion.py     # Ground motion generation
├── postprocess.py       # Response analysis
├── plotting.py          # Visualization functions
├── report.py            # PDF report generation
├── ui_tabs.py           # UI tab components
├── design_codes.py      # IS 1893 code provisions
├── styles.py            # CSS styles
│
├── exceptions.py        # Custom exception hierarchy
├── validators.py        # Input validation utilities
├── logging_config.py    # Production logging setup
├── production_config.py # Environment configuration
├── health.py            # Health check utilities
│
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## 🔧 Architecture

### Exception Hierarchy

```
SeismicAppError (base)
├── ValidationError      # Input validation failures
├── MatrixError          # Matrix assembly/operation issues
├── SolverError          # Time integration failures
│   ├── ConvergenceError # Newton-Raphson non-convergence
│   └── NumericalInstabilityError # NaN/Inf detection
├── GroundMotionError    # Ground motion generation issues
└── ReportGenerationError # PDF generation failures
```

### Key Components

1. **Structural Analysis**
   - Build mass, stiffness, damping matrices
   - Support for base isolation and TMD
   - Rayleigh damping with modal targets

2. **Time History Solvers**
   - Newmark-β linear solver (β=0.25, γ=0.5)
   - Newmark-β nonlinear solver with Bouc-Wen hysteresis
   - Bidirectional ground motion support

3. **Post-Processing**
   - Energy balance calculation
   - Response spectrum computation
   - Fragility curve generation
   - Performance level assessment

4. **Validation Layer**
   - Input parameter bounds checking
   - Matrix positive-definiteness verification
   - NaN/Inf detection in results

## 📊 Features

- **Linear Analysis**: Elastic response under earthquake excitation
- **Nonlinear Analysis**: Hysteretic behavior with ductility tracking
- **Control Devices**: Base isolation, TMD, viscous dampers
- **Ground Motion**: Synthetic (Kanai-Tajimi, Clough-Penzien) and scaled records
- **Response Spectra**: Ground and floor response spectra
- **Performance Assessment**: IO, LS, CP performance levels
- **Code Compliance**: IS 1893:2016 provisions

## 🧪 Testing

```bash
# Run health check self-tests
python health.py

# Install dev dependencies (optional)
pip install pytest black mypy flake8

# Run tests (if available)
pytest tests/

# Format code
black .

# Type checking
mypy .
```

## 📝 License

MIT License - For educational purposes.

## ⚠️ Disclaimer

This tool is for **educational purposes only**. Results should not be used for actual structural design without proper engineering review and validation against established software.
