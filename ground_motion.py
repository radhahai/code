"""
ground_motion.py - Earthquake Ground Motion Generation

This module handles the generation of synthetic ground motions
matching various spectral characteristics for educational purposes.

IMPORTANT: All ground motions are SYNTHETICALLY GENERATED.
They are designed to match target spectral characteristics
but are NOT actual recorded earthquake data.

Features:
- Kanai-Tajimi spectrum-based generation
- Historical earthquake spectrum matching
- Educational waveforms (sine, Ricker, sweep)
- Quiet period for residual drift calculation
- Comprehensive validation
"""

import numpy as np
from typing import Tuple, Optional
from scipy.signal import butter, filtfilt
import warnings

from models import SeismicEvent, EARTHQUAKE_DATABASE
from exceptions import GroundMotionError, ValidationError
from logging_config import get_logger

logger = get_logger("ground_motion")

# Constants
MAX_PGA = 5.0  # g - Physical limit
MAX_DURATION = 600.0  # seconds
MIN_DT = 1e-4  # seconds


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_ground_motion(
    event: SeismicEvent,
    pga: float,
    duration: float,
    dt: float,
    soil_type: str,
    scale_factor: float = 1.0,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Master ground motion generator supporting multiple earthquake types.
    
    IMPORTANT: All motions are SYNTHETICALLY GENERATED to match
    target spectral characteristics. They are NOT recorded data.
    
    Parameters
    ----------
    event : SeismicEvent
        Type of ground motion to generate
    pga : float
        Peak ground acceleration in g
    duration : float
        Duration in seconds
    dt : float
        Time step in seconds
    soil_type : str
        Soil type: "soft", "medium", "stiff", "rock"
    scale_factor : float
        Amplitude scaling factor
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    t : np.ndarray
        Time array in seconds (includes quiet period at end)
    ag : np.ndarray
        Ground acceleration in m/s²
    
    Raises
    ------
    ValidationError
        If input parameters are out of valid range
    GroundMotionError
        If generation fails
    """
    # Validate inputs
    if pga <= 0:
        raise ValidationError("pga", pga, "> 0")
    if pga > MAX_PGA:
        logger.warning(f"PGA {pga}g exceeds physical maximum, clamping to {MAX_PGA}g")
        pga = MAX_PGA
    
    if duration <= 0:
        raise ValidationError("duration", duration, "> 0")
    if duration > MAX_DURATION:
        logger.warning(f"Duration {duration}s exceeds limit, clamping to {MAX_DURATION}s")
        duration = MAX_DURATION
    
    if dt <= 0 or dt < MIN_DT:
        raise ValidationError("dt", dt, f">= {MIN_DT}")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    logger.debug(f"Generating {event.value}: PGA={pga}g, duration={duration}s, dt={dt}s")
    
    # Add quiet period at end for free vibration decay (improves residual drift accuracy)
    quiet_period = min(5.0, duration * 0.25)  # 25% of duration or 5s max
    total_duration = duration + quiet_period
    
    t = np.arange(0, total_duration, dt)
    n_motion = int(duration / dt)
    n_total = len(t)
    
    # Generate time array for motion portion only
    t_motion = np.arange(0, duration, dt)
    n = len(t_motion)
    
    # Generate base motion based on event type
    if event == SeismicEvent.KANAI_TAJIMI:
        ag = _kanai_tajimi_spectrum(n, dt, pga, soil_type)
    elif event == SeismicEvent.CUSTOM_SINE:
        ag = _custom_sine_pulse(t_motion, pga)
    elif event == SeismicEvent.RICKER_WAVELET:
        ag = _ricker_wavelet(t_motion, pga, duration)
    elif event == SeismicEvent.HARMONIC_SWEEP:
        ag = _harmonic_sweep(t_motion, pga, duration)
    elif event == SeismicEvent.MULTI_PULSE:
        ag = _multi_frequency_pulse(t_motion, pga)
    else:
        # Synthetic records based on database characteristics
        ag = _synthetic_historical(event, n, dt, pga, soil_type)
    
    # Apply envelope
    ag = _apply_envelope(ag, t_motion, duration)
    
    # Scale to target PGA
    max_ag = np.max(np.abs(ag))
    if max_ag > 0:
        ag = ag / max_ag * (pga * 9.81) * scale_factor
    
    # Apply baseline correction
    ag = _baseline_correction(ag, dt)
    
    # Append quiet period (zero acceleration) for free vibration decay
    # This improves residual drift calculation accuracy
    n_quiet = n_total - len(ag)
    if n_quiet > 0:
        ag = np.concatenate([ag, np.zeros(n_quiet)])
    
    return t, ag


def generate_bidirectional_motion(
    event: SeismicEvent,
    pga: float,
    duration: float,
    dt: float,
    soil_type: str,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate correlated X and Y components of ground motion.
    
    Parameters
    ----------
    event : SeismicEvent
        Type of ground motion
    pga : float
        Peak ground acceleration for X component in g
    duration : float
        Duration in seconds
    dt : float
        Time step in seconds
    soil_type : str
        Soil type
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    t : np.ndarray
        Time array
    ag_x : np.ndarray
        X-component acceleration in m/s²
    ag_y : np.ndarray
        Y-component acceleration in m/s²
    """
    # Generate X component
    t, ag_x = generate_ground_motion(event, pga, duration, dt, soil_type, 1.0, random_seed)
    
    # Y component typically 0.85 of X with phase shift and different seed
    if random_seed is not None:
        np.random.seed(random_seed + 1000)
    
    _, ag_y = generate_ground_motion(event, pga * 0.85, duration, dt, soil_type, 1.0, None)
    
    # Small time delay for Y component
    shift = int(0.05 / dt)
    ag_y = np.roll(ag_y, shift)
    
    return t, ag_x, ag_y


# =============================================================================
# SPECTRUM-BASED GENERATION
# =============================================================================

def _kanai_tajimi_spectrum(n: int, dt: float, pga: float, soil_type: str) -> np.ndarray:
    """
    Generate ground motion using Kanai-Tajimi filtered white noise.
    
    The Kanai-Tajimi spectrum models the frequency content of
    earthquake ground motions through site soil characteristics.
    """
    # Soil-dependent parameters
    params = {
        "soft":   {"wg": 5.0,  "zeta": 0.6},
        "medium": {"wg": 10.0, "zeta": 0.5},
        "stiff":  {"wg": 15.0, "zeta": 0.4},
        "rock":   {"wg": 25.0, "zeta": 0.3},
    }
    
    wg = params.get(soil_type, params["medium"])["wg"]
    zeta = params.get(soil_type, params["medium"])["zeta"]
    
    # Generate white noise
    white = np.random.normal(0, 1, n)
    
    # Frequency array
    freq = np.fft.rfftfreq(n, dt)
    w = 2 * np.pi * freq
    w[0] = 1e-6  # Avoid division by zero
    
    # Kanai-Tajimi transfer function with Clough-Penzien high-pass modification
    wf = 0.1 * wg  # High-pass filter frequency
    zetaf = 0.6
    
    # Kanai-Tajimi filter
    H_kt = np.sqrt((1 + 4 * zeta**2 * (w / wg)**2) / 
                   ((1 - (w / wg)**2)**2 + 4 * zeta**2 * (w / wg)**2))
    
    # Clough-Penzien high-pass filter (removes low-frequency drift)
    H_hp = (w / wf)**2 / np.sqrt((1 - (w / wf)**2)**2 + 4 * zetaf**2 * (w / wf)**2)
    
    # Combined transfer function
    H = H_kt * H_hp
    
    # Apply filter in frequency domain
    white_fft = np.fft.rfft(white)
    ag = np.fft.irfft(white_fft * H, n)
    
    return ag


def _synthetic_historical(
    event: SeismicEvent,
    n: int,
    dt: float,
    pga: float,
    soil_type: str
) -> np.ndarray:
    """
    Generate synthetic record mimicking historical earthquake characteristics.
    
    These are NOT actual recorded motions, but synthetically generated
    to match the spectral characteristics of historical events.
    """
    info = EARTHQUAKE_DATABASE.get(event, {"pga": 0.4, "freq_content": "broadband"})
    
    # Base frequency characteristics based on event type
    freq_content = info.get("freq_content", "broadband")
    
    if freq_content == "near_field":
        # Near-field: higher frequencies, pulse-like
        wg = 12.0
        pulse_factor = 0.3
    elif freq_content == "soft_soil":
        # Soft soil: low frequency amplification
        wg = 3.0
        pulse_factor = 0.1
    elif freq_content == "long_duration":
        # Subduction: longer duration, complex
        wg = 8.0
        pulse_factor = 0.15
    else:
        # Broadband
        wg = 10.0
        pulse_factor = 0.2
    
    # Generate base motion using modified Kanai-Tajimi
    ag = _kanai_tajimi_spectrum(n, dt, pga, soil_type)
    
    # Add directivity pulse for near-field events
    if pulse_factor > 0.2:
        t = np.arange(n) * dt
        Tp = 1.0 + 0.5 * np.random.random()
        t_pulse = n * dt * 0.3
        pulse = pulse_factor * np.sin(2 * np.pi / Tp * (t - t_pulse)) * \
                np.exp(-((t - t_pulse) / Tp)**2)
        ag = ag + pulse * np.max(np.abs(ag))
    
    return ag


# =============================================================================
# SIMPLE WAVEFORMS (Educational)
# =============================================================================

def _custom_sine_pulse(t: np.ndarray, pga: float) -> np.ndarray:
    """Simple sine pulse for testing and education."""
    T = 1.0  # Period
    n_cycles = 5
    duration = n_cycles * T
    
    ag = np.zeros_like(t)
    mask = t < duration
    ag[mask] = np.sin(2 * np.pi / T * t[mask]) * np.sin(np.pi * t[mask] / duration)
    
    return ag


def _ricker_wavelet(t: np.ndarray, pga: float, duration: float) -> np.ndarray:
    """
    Ricker wavelet (Mexican hat) for impulsive loading.
    Commonly used in geophysics and seismic testing.
    """
    fp = 2.0  # Peak frequency
    t0 = duration * 0.3
    tau = (t - t0) * 2 * np.pi * fp
    ag = (1 - 2 * tau**2) * np.exp(-tau**2)
    
    return ag


def _harmonic_sweep(t: np.ndarray, pga: float, duration: float) -> np.ndarray:
    """Frequency sweep from low to high frequency."""
    f0, f1 = 0.5, 15.0
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    ag = np.sin(phase)
    
    return ag


def _multi_frequency_pulse(t: np.ndarray, pga: float) -> np.ndarray:
    """Multi-frequency pulse for broadband testing."""
    ag = np.zeros_like(t)
    frequencies = [0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 12.0]
    amplitudes = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
    
    for f, a in zip(frequencies, amplitudes):
        ag += a * np.sin(2 * np.pi * f * t + np.random.random() * np.pi)
    
    return ag


# =============================================================================
# POST-PROCESSING
# =============================================================================

def _apply_envelope(ag: np.ndarray, t: np.ndarray, duration: float) -> np.ndarray:
    """
    Apply Saragoni-Hart envelope function.
    
    This shapes the ground motion with realistic build-up,
    strong motion phase, and decay.
    """
    t_norm = t / duration
    alpha = 0.15 + 0.1 * np.random.random()
    epsilon = (t_norm / alpha)**2 * np.exp(1 - (t_norm / alpha)**2)
    
    return ag * epsilon


def _baseline_correction(ag: np.ndarray, dt: float) -> np.ndarray:
    """
    Remove low-frequency drift using high-pass filter.
    
    This ensures zero mean velocity and displacement at the end
    of the ground motion record.
    """
    try:
        nyq = 0.5 / dt
        low = 0.1 / nyq
        if low >= 1.0:
            low = 0.9
        b, a = butter(4, low, btype='high')
        ag_filtered = filtfilt(b, a, ag)
        return ag_filtered
    except Exception:
        return ag


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_arias_intensity(ag: np.ndarray, dt: float) -> float:
    """
    Compute Arias Intensity of ground motion.
    
    Ia = (π / 2g) * ∫ a(t)² dt
    
    Parameters
    ----------
    ag : np.ndarray
        Ground acceleration in m/s²
    dt : float
        Time step
    
    Returns
    -------
    float
        Arias Intensity in m/s
    """
    g = 9.81
    return np.pi / (2 * g) * np.sum(ag**2) * dt


def compute_significant_duration(ag: np.ndarray, dt: float, low: float = 0.05, high: float = 0.95) -> float:
    """
    Compute significant duration (D5-95 or custom).
    
    Parameters
    ----------
    ag : np.ndarray
        Ground acceleration in m/s²
    dt : float
        Time step
    low : float
        Lower bound of Arias intensity fraction (default 5%)
    high : float
        Upper bound of Arias intensity fraction (default 95%)
    
    Returns
    -------
    float
        Significant duration in seconds
    """
    # Cumulative Arias intensity
    husid = np.cumsum(ag**2) * dt
    husid_norm = husid / husid[-1]
    
    # Find times at low and high percentages
    t_low = np.argmax(husid_norm >= low) * dt
    t_high = np.argmax(husid_norm >= high) * dt
    
    return t_high - t_low
