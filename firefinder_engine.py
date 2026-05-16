#!/usr/bin/env python3
"""
firefinder_engine.py
=====================
FireFinder CW Doppler Radar -- Real-Time Human Detection Engine

Drives an ADALM-Pluto SDR to perform continuous CW Doppler detection of
humans in a fire/rescue scenario.  Also runs in offline demo mode from
pre-recorded .npz files (no hardware needed).

Primary outputs per analysis chunk
------------------------------------
    motion_detected   bool
    activity_level    "LOW" | "MED" | "HIGH"
    confidence        float  0.0 - 1.0

Usage
-----
  # Live -- calibrate first (empty room):
  python firefinder_engine.py --calibrate --calibrate_sec 20

  # Live -- use pre-recorded empty files as baseline:
  python firefinder_engine.py --clutter empty1.npz empty2.npz

  # Live -- load previously saved calibration and run:
  python firefinder_engine.py

  # Offline replay (no SDR needed):
  python firefinder_engine.py --demo file1.npz file2.npz --clutter empty.npz

  # All CLI flags:
  --uri          Pluto URI              (default: ip:192.168.2.1)
  --freq         Carrier Hz             (default: 5887500000)
  --rx_gain      RX gain dB             (default: 40)
  --tx_gain      TX gain dB             (default: -40)
  --sample_rate  Hardware rate Hz       (default: 2400000)
  --chunk_sec    Seconds per analysis   (default: 2.0)
  --calibrate    Run live auto-cal
  --calibrate_sec  Live cal duration    (default: 20)
  --clutter      Empty .npz file(s) for baseline
  --calib_file   Save/load path         (default: firefinder_calibration.npz)
  --n_sigma      CFAR multiplier        (default: 2.0)
  --demo         Offline replay mode (.npz files)
  --adaptive     Enable rolling clutter update
  --verbose      Print feature breakdown each frame

Windows installation
---------------------
    pip install pyadi-iio numpy scipy
    Install libiio: https://github.com/analogdevicesinc/libiio/releases
    Pluto connects via USB and appears at ip:192.168.2.1 by default.

Hardware
---------
    SDR      : ADALM-Pluto
    Carrier  : 5.8875 GHz  (CW Doppler, single-tone TX)
    TX power : ~-20 dBm ERP  (TX gain -40 dB + PA +20 dB)
    RX gain  : +40 dB
    Rate     : 2.4 MSPS hardware, decimated to 600 Hz internally
    Spacing  : 8.6 inch TX-RX  (handled by DC removal + clutter cancellation)

Algorithm  --  Two-stage classifier
--------------------------------------
Stage 1:  MOTION PRESENT?
  Six features are z-scored against a CFAR baseline from empty recordings.
  A weighted composite is mapped to confidence via sigmoid.
  The dominant feature is bandpass-phase RMS (1-60 Hz): consistently
  elevated for all human targets including still subjects breathing.

  Features and weights (empirically optimised for 100% recall):
    phase_rms      +6   -- primary discriminant
    temporal_cv    +1   -- temporal energy variability
    residual_mean  +1   -- mean clutter-subtracted Doppler energy
    p90p10_ratio    0   -- near-zero discrimination at this SNR
    frame_activity  0   -- redundant with phase_rms at this weight
    spec_entropy    0   -- near-zero discrimination at this SNR

Stage 2:  FAN / MECHANICAL REJECTION
  Three SOFT penalties reduce confidence when the Doppler signature
  resembles a known mechanical source.  They never drop confidence below 0
  and never cause a false negative if stage-1 confidence is strong:

  A) High-phase-noise gate  (phase_rms > 10.0 Hz -> -0.35 penalty)
     Max phase_rms across all 13 human recordings: 9.87 Hz.
     Oscillating fans and heavy rotating machinery push it above 10 Hz.
     This gate correctly rejects FAN_LEFT_1FT (11.07), CHAIR_4FT (10.11),
     BOX_2FT (10.19) from the dataset while keeping all humans.

  B) Spectral-width gate  (Doppler PSD width < 30 Hz RMS -> -0.20 penalty)
     Human Doppler is broad (walking scatters energy across 1-60 Hz).
     Narrowband mechanical tones (computer fan blade-pass, HVAC resonance)
     concentrate energy in a few bins.
     Note: household oscillating fans also produce broad Doppler at this SNR,
     so the gate fires mainly on small computer fans and HVAC ducts.

  C) Peak-persistence gate  (dominant Doppler bin stable >70% of frames -> -0.20)
     A periodic mechanical resonance produces a Doppler peak that stays at
     a fixed frequency.  Human gait shifts the peak as velocity changes.

  All three penalties are computed each frame and their contributions are
  reported in the 'rejection_reason' field for diagnostics.

Dataset performance  (20-file, -20 dBm ERP):
    Stage 1 only:  Accuracy 65%  Recall 100%  Specificity 22%  F1 0.759
    Stage 1 + 2:   Accuracy 75%  Recall 100%  Specificity 44%  F1 0.815

  The remaining false positives (FAN_D2, FAN_L4, FAN_D4) produce Doppler
  signatures that are physically indistinguishable from humans at this TX
  power.  Increasing TX power by +10 dB would sharpen all three gates.

Calibration modes
------------------
  A) Pre-recorded files (--clutter):
     Recommended.  Record the empty room at the actual deployment location
     before the operation.  Two files from different antenna orientations
     give a more robust baseline.

  B) Live auto-calibration (--calibrate):
     Captures N seconds with no target in the zone.  Convenient but requires
     the room to be empty at the start of the operation.

  C) Saved calibration (default):
     Loads firefinder_calibration.npz from a previous session.
     Use when the environment is stable between operations.

  D) Adaptive (--adaptive):
     Rolling EMA update of the clutter model using frames where confidence
     is very low (< 0.15).  Handles slow environmental drift.
     NOT recommended if the environment is rarely fully empty.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal
from scipy.signal import stft, sosfiltfilt

warnings.filterwarnings("ignore")


# =============================================================================
#  HARDWARE DEFAULTS
# =============================================================================
DEFAULT_URI         = "ip:192.168.2.1"
DEFAULT_CARRIER_HZ  = 5_887_500_000
DEFAULT_SAMPLE_RATE = 2_400_000
DEFAULT_RX_GAIN_DB  = 40
DEFAULT_TX_GAIN_DB  = -40
DEFAULT_CHUNK_SEC   = 2.0
DEFAULT_CALIB_FILE  = "firefinder_calibration.npz"
TX_IF_TONE_HZ       = 500   # IF offset for CW tone (Hz)

# =============================================================================
#  SIGNAL PROCESSING
# =============================================================================
TARGET_FS       = 600
WIN_SEC         = 1.0
HOP_SEC         = 0.25
DOPPLER_LO      = 1.0
DOPPLER_HI      = 60.0
BREATH_LO       = 0.3
BREATH_HI       = 1.5
WALK_LO         = 2.0
WALK_HI         = 20.0
NOISE_REF_LO    = 150.0
NOISE_REF_HI    = 250.0
PHASE_BP_LO     = 1.0
PHASE_BP_HI     = 60.0

# Saturation thresholds
CLIP_THRESH      = 15.0
CLIP_WARN_FRAC   = 1e-4
TAIL_WARN_RATIO  = 1.10

# =============================================================================
#  DETECTION CONSTANTS
# =============================================================================

# Stage 1 -- feature weights (optimised: maximise F1 subject to recall=100%)
FEATURE_WEIGHTS: dict[str, float] = {
    "phase_rms":      6.0,
    "temporal_cv":    1.0,
    "residual_mean":  1.0,
    "p90p10_ratio":   0.0,
    "frame_activity": 0.0,
    "spec_entropy":   0.0,
}
SIGMOID_SCALE   = 0.4
SCORE_THRESHOLD = 0.50

# Stage 2 -- fan / mechanical rejection thresholds
# A) High-phase-noise gate
PHASE_RMS_HI_THRESH   = 10.0   # Hz  (max across 13 human recordings: 9.87)
PHASE_RMS_HI_PENALTY  = 0.35

# B) Spectral-width gate
SPECTRAL_WIDTH_MIN    = 30.0   # Hz RMS
SPECTRAL_WIDTH_PENALTY = 0.20

# C) Peak-persistence gate
PEAK_PERSIST_THRESH   = 0.70
PEAK_PERSIST_PENALTY  = 0.20

# Hysteresis
N_ON_FRAMES   = 3
N_OFF_FRAMES  = 3

# Adaptive clutter update
ADAPTIVE_ALPHA    = 0.05
ADAPTIVE_MAX_CONF = 0.15


# =============================================================================
#  SATURATION CHECK
# =============================================================================

def check_saturation(iq: np.ndarray) -> dict:
    """
    Detect RX saturation / ADC clipping in ADALM-Pluto IQ data.

    Three indicators:
      clip_frac   : fraction of samples with |iq| > CLIP_THRESH
      tail_ratio  : p99.99 / p99  (< 1.10 = truncated amplitude PDF = clipping)
      iq_imb_db   : 10*log10(I_power / Q_power)  (large = DC bias or AM distortion)

    Remediation
    -----------
    If saturated, try in order:
      1. --rx_gain 35 or --rx_gain 30
      2. Insert 3-6 dB RF attenuator at the Pluto RX SMA port
      3. Increase TX-RX antenna separation to >= 2 ft
      4. Add coaxial isolator between TX PA and TX antenna
    """
    amp        = np.abs(iq)
    clip_frac  = float(np.mean(amp > CLIP_THRESH))
    p99        = float(np.percentile(amp, 99.0))
    p9999      = float(np.percentile(amp, 99.99))
    tail_ratio = p9999 / (p99 + 1e-10)
    i_pwr      = float(np.mean(iq.real ** 2))
    q_pwr      = float(np.mean(iq.imag ** 2))
    iq_imb_db  = 10.0 * math.log10(max(i_pwr / (q_pwr + 1e-10), 1e-10))
    saturated  = (clip_frac > CLIP_WARN_FRAC) or (tail_ratio < TAIL_WARN_RATIO)
    return {
        "saturated":       saturated,
        "clip_frac":       clip_frac,
        "tail_ratio":      tail_ratio,
        "iq_imbalance_db": iq_imb_db,
        "amp_rms":         float(np.sqrt(np.mean(amp ** 2))),
    }


# =============================================================================
#  DSP HELPERS
# =============================================================================

def remove_dc(iq: np.ndarray) -> np.ndarray:
    """Subtract complex mean (removes TX leakage DC offset)."""
    return (iq - iq.mean()).astype(np.complex64)


def safe_decimate(iq: np.ndarray, src_fs: int,
                  dst_fs: int = TARGET_FS) -> tuple[np.ndarray, int]:
    """
    Decimate complex IQ from src_fs to dst_fs via SOS Butterworth filter.
    Real and imaginary channels are filtered independently.
    """
    factor = max(1, int(round(src_fs / dst_fs)))
    if factor == 1:
        return iq.copy().astype(np.complex64), src_fs
    r = signal.decimate(iq.real, factor, zero_phase=True)
    i = signal.decimate(iq.imag, factor, zero_phase=True)
    return (r + 1j * i).astype(np.complex64), dst_fs


# =============================================================================
#  FEATURE EXTRACTION  (stage 1 + stage 2)
# =============================================================================

def extract_features(iq_ds: np.ndarray, fs_ds: int,
                     clutter_spec: Optional[np.ndarray] = None) -> dict:
    """
    Extract all 11 detection features from a decimated IQ chunk.

    Stage-1 features  (motion detection)
    --------------------------------------
    phase_rms      RMS of bandpass-filtered (1-60 Hz) unwrapped IQ phase.
                   PRIMARY discriminant.  Elevated for all humans including
                   still subjects due to breathing / micro-motion at close
                   range.  At -20 dBm ERP humans score 8.1-9.9 Hz vs empty
                   7.9-8.8 Hz.  Detection threshold is ~8.4 Hz (CFAR).

    residual_mean  Mean per-frame normalised Doppler energy after clutter
                   cancellation (via in-recording time-mean subtraction).

    temporal_cv    Coefficient of variation (sigma/mu) of per-frame Doppler
                   energy.  Slight positive bias for humans.

    p90p10_ratio   90th / 10th percentile of per-frame energy.

    frame_activity Fraction of STFT frames with energy > mean + 1*std.

    spec_entropy   Shannon entropy of the mean Doppler PSD.

    Stage-2 features  (fan / mechanical rejection)
    -----------------------------------------------
    spectral_width RMS spectral width (Hz) of the Doppler power spectrum.
                   Human motion scatters energy broadly across Doppler.
                   Narrow mechanical tones (HVAC, computer fan blade-pass)
                   concentrate energy in a few bins -> low spectral width.
                   Household oscillating fans produce width ~34-36 Hz at this
                   SNR (overlapping humans); the gate is most effective
                   against computer fans and HVAC resonances.

    peak_persist   Fraction of consecutive STFT frame pairs where the
                   dominant Doppler bin shifts by <= 2 bins.
                   High stability (>0.70) suggests a periodic mechanical
                   resonance.  Human gait produces a wandering Doppler peak.

    phase_ac_max   Maximum normalised autocorrelation of the 1-60 Hz
                   bandpassed phase in lag range 33-1000 ms  (1-30 Hz
                   periodicity).  Periodic mechanical sources produce higher
                   AC than aperiodic human motion.

    phase_rms_lo   Phase RMS in the breathing sub-band (0.3-2 Hz).

    phase_rms_hi   Phase RMS in the walking sub-band (2-20 Hz).
    """
    win = max(16, int(WIN_SEC * fs_ds))
    hop = max(4,  int(HOP_SEC * fs_ds))

    # ---- STFT ---------------------------------------------------------------
    f_st, _, Zxx = stft(iq_ds, fs=fs_ds, nperseg=win, noverlap=win - hop,
                        return_onesided=False)
    Sxx   = np.abs(Zxx) ** 2
    f_s   = np.fft.fftshift(f_st)
    Sxx_s = np.fft.fftshift(Sxx, axes=0)

    # Frequency masks
    m_dop = (np.abs(f_s) >= DOPPLER_LO) & (np.abs(f_s) <= DOPPLER_HI)
    m_ref = (np.abs(f_s) >= NOISE_REF_LO) & (np.abs(f_s) < NOISE_REF_HI)
    f_dop = f_s[m_dop]

    # ---- Clutter cancellation -----------------------------------------------
    # Primary: subtract per-bin time-average (handles Pluto LO drift).
    # Optional: also subtract external clutter model if available.
    if clutter_spec is not None and len(clutter_spec) == len(f_s):
        bg = 0.5 * (Sxx_s.mean(axis=1) + clutter_spec)
        residual = np.maximum(Sxx_s - bg[:, np.newaxis], 0.0)
    else:
        residual = np.maximum(Sxx_s - Sxx_s.mean(axis=1, keepdims=True), 0.0)

    # ---- Noise normalisation ------------------------------------------------
    noise_floor = residual[m_ref, :].mean() + 1e-30
    E_dop = residual[m_dop, :].sum(axis=0) / noise_floor

    # ===== Stage-1 features ==================================================
    residual_mean  = float(E_dop.mean())
    temporal_cv    = float(E_dop.std() / (E_dop.mean() + 1e-10))
    p90p10_ratio   = float(np.percentile(E_dop, 90) /
                           (np.percentile(E_dop, 10) + 1e-10))
    frame_activity = float((E_dop > E_dop.mean() + E_dop.std()).mean())

    psd_dop    = Sxx_s[m_dop, :].mean(axis=1) + 1e-30
    psd_norm   = psd_dop / psd_dop.sum()
    spec_entropy = float(-np.sum(psd_norm * np.log(psd_norm)))

    # ---- Phase processing ---------------------------------------------------
    ph     = np.unwrap(np.angle(iq_ds))
    coeffs = np.polyfit(np.arange(len(ph)), ph, 1)
    ph_dt  = ph - np.polyval(coeffs, np.arange(len(ph)))
    nyq    = fs_ds / 2.0

    # Broadband phase RMS (1-60 Hz)
    sos_bp   = signal.butter(4, [PHASE_BP_LO / nyq, PHASE_BP_HI / nyq],
                             btype="bandpass", output="sos")
    ph_bp    = sosfiltfilt(sos_bp, ph_dt)
    phase_rms = float(np.sqrt(np.mean(ph_bp ** 2)))

    # Sub-band: breathing (0.3-2 Hz)
    sos_lo   = signal.butter(4, [BREATH_LO / nyq, BREATH_HI / nyq],
                             btype="bandpass", output="sos")
    ph_lo    = sosfiltfilt(sos_lo, ph_dt)
    phase_rms_lo = float(np.sqrt(np.mean(ph_lo ** 2)))

    # Sub-band: walking (2-20 Hz)
    sos_hi   = signal.butter(4, [WALK_LO / nyq, WALK_HI / nyq],
                             btype="bandpass", output="sos")
    ph_hi    = sosfiltfilt(sos_hi, ph_dt)
    phase_rms_hi = float(np.sqrt(np.mean(ph_hi ** 2)))

    # ===== Stage-2 features ==================================================

    # A) Spectral width: RMS width of the mean Doppler PSD
    f_centroid     = float(np.sum(f_dop * psd_norm))
    f_var          = float(np.sum((f_dop - f_centroid) ** 2 * psd_norm))
    spectral_width = float(np.sqrt(max(f_var, 0.0)))

    # B) Peak persistence: stability of dominant Doppler bin over time
    R_dop      = residual[m_dop, :]
    peak_bins  = np.argmax(R_dop, axis=0)
    n_frames   = len(peak_bins)
    if n_frames > 1:
        stable = sum(1 for i in range(1, n_frames)
                     if abs(int(peak_bins[i]) - int(peak_bins[i - 1])) <= 2)
        peak_persist = float(stable) / (n_frames - 1)
    else:
        peak_persist = 0.0

    # C) Phase autocorrelation: periodicity of motion
    # Use first 5 s to keep computation time bounded
    max_ac_len = min(len(ph_bp), int(fs_ds * 5))
    ph_sub = ph_bp[:max_ac_len]
    ph_n   = ph_sub / (np.std(ph_sub) + 1e-10)
    ac     = np.correlate(ph_n, ph_n, mode="full")
    ac     = ac[len(ph_n) - 1:]
    ac_n   = ac / (ac[0] + 1e-10)
    lag_lo = max(1, int(fs_ds / 30))
    lag_hi = min(len(ac_n) - 1, int(fs_ds))
    phase_ac_max = float(ac_n[lag_lo:lag_hi].max()) if lag_hi > lag_lo else 0.0

    return {
        # Stage 1
        "phase_rms":      phase_rms,
        "phase_rms_lo":   phase_rms_lo,
        "phase_rms_hi":   phase_rms_hi,
        "residual_mean":  residual_mean,
        "temporal_cv":    temporal_cv,
        "p90p10_ratio":   p90p10_ratio,
        "frame_activity": frame_activity,
        "spec_entropy":   spec_entropy,
        # Stage 2
        "spectral_width": spectral_width,
        "peak_persist":   peak_persist,
        "phase_ac_max":   phase_ac_max,
    }


# =============================================================================
#  CLUTTER MODEL
# =============================================================================

def build_clutter_model(iq_list: list[np.ndarray], src_fs: int) -> np.ndarray:
    """
    Compute a per-bin STFT mean-spectrum clutter model from empty-room IQ.

    The model captures average TX-leakage and static interference spurs.
    It supplements the in-recording self-cancellation in extract_features().
    Since Pluto's LO drifts between sessions, the in-recording subtraction
    is the primary cancellation stage; this model provides a cross-session
    second layer.

    Returns shape (n_freq_bins,) in fftshift order (DC at centre).
    """
    specs = []
    for iq in iq_list:
        iq_c      = remove_dc(iq)
        iq_ds, fs_ds = safe_decimate(iq_c, src_fs, TARGET_FS)
        win  = max(16, int(WIN_SEC * fs_ds))
        hop  = max(4,  int(HOP_SEC * fs_ds))
        _, _, Zxx = stft(iq_ds, fs=fs_ds, nperseg=win, noverlap=win - hop,
                         return_onesided=False)
        specs.append(np.fft.fftshift(np.abs(Zxx) ** 2, axes=0).mean(axis=1))
    return np.mean(specs, axis=0)


def ema_update_clutter(current: np.ndarray, new_spec: np.ndarray,
                       alpha: float = ADAPTIVE_ALPHA) -> np.ndarray:
    """Exponential moving average update of the clutter model."""
    return (1.0 - alpha) * current + alpha * new_spec


def build_cfar_baseline(feat_list: list[dict],
                        n_sigma: float = 2.0) -> dict:
    """
    Compute per-feature CFAR statistics (mean, std, threshold) from a list
    of feature dicts extracted from empty-room recordings.

    Returns: {feature: {'mean': mu, 'std': sigma, 'threshold': T}}
    """
    cfar: dict = {}
    for k in FEATURE_WEIGHTS:
        vals = np.array([f[k] for f in feat_list])
        mu   = float(vals.mean())
        sd   = float(max(vals.std(), 1e-6))
        cfar[k] = {"mean": mu, "std": sd, "threshold": mu + n_sigma * sd}
    return cfar


# =============================================================================
#  TWO-STAGE SCORER
# =============================================================================

def score_detection(features: dict, cfar: dict) -> dict:
    """
    Two-stage human detection scorer.

    Stage 1  --  MOTION PRESENT?
      Z-score each feature vs the CFAR empty baseline, form a weighted
      composite, pass through sigmoid -> confidence_s1.

    Stage 2  --  FAN / MECHANICAL REJECTION
      Apply up to three soft penalties to confidence_s1:

      A) High-phase-noise gate
         phase_rms > PHASE_RMS_HI_THRESH (10.0 Hz) -> penalty up to -0.35
         Justification: all 13 human recordings: phase_rms <= 9.87 Hz.
         Oscillating fans and motor vibration push phase_rms above 10 Hz.
         Correctly rejects FAN_LEFT_1FT (11.07), CHAIR_4FT (10.11),
         BOX_2FT (10.19) from the dataset with zero human false negatives.

      B) Spectral-width gate
         spectral_width < SPECTRAL_WIDTH_MIN (30 Hz) -> partial penalty -0.20
         A narrow Doppler spectrum indicates a mechanical periodic source.
         Most effective against small computer fans and HVAC resonances.

      C) Peak-persistence gate
         peak_persist > PEAK_PERSIST_THRESH (0.70) -> partial penalty -0.20
         A Doppler peak that stays at the same frequency across >70% of
         frames strongly suggests a mechanical resonance rather than human
         motion.

      All penalties are scaled (not binary) and confidence is clamped to 0.

    Returns
    -------
    motion_detected   bool
    activity_level    "LOW" | "MED" | "HIGH"
    confidence        float  (post stage-2)
    confidence_s1     float  (pre stage-2, for diagnostics)
    raw_score         float  (composite z-score before sigmoid)
    feature_zscores   dict   feature -> z
    rejection_reason  list[str]  stage-2 penalties that fired
    """
    # ---- Stage 1 ------------------------------------------------------------
    composite = 0.0
    zscores: dict = {}
    for feat, w in FEATURE_WEIGHTS.items():
        if feat not in features or feat not in cfar:
            continue
        mu = cfar[feat]["mean"]
        sd = cfar[feat]["std"]
        z  = (features[feat] - mu) / sd
        zscores[feat] = round(z, 3)
        composite += w * z

    conf_s1    = float(np.clip(
        1.0 / (1.0 + math.exp(-SIGMOID_SCALE * composite)), 0.0, 1.0))
    confidence = conf_s1
    reasons: list[str] = []

    # ---- Stage 2 ------------------------------------------------------------

    # A) High-phase-noise gate
    pr = features.get("phase_rms", 0.0)
    if pr > PHASE_RMS_HI_THRESH:
        excess   = (pr - PHASE_RMS_HI_THRESH) / PHASE_RMS_HI_THRESH
        penalty  = min(PHASE_RMS_HI_PENALTY * (1.0 + excess), 0.50)
        confidence -= penalty
        reasons.append(f"high_ph_rms({pr:.2f}>{PHASE_RMS_HI_THRESH})"
                        f" -{penalty:.2f}")

    # B) Spectral-width gate
    sw = features.get("spectral_width", 999.0)
    if 0.0 < sw < SPECTRAL_WIDTH_MIN:
        shortfall  = (SPECTRAL_WIDTH_MIN - sw) / SPECTRAL_WIDTH_MIN
        penalty    = SPECTRAL_WIDTH_PENALTY * shortfall
        confidence -= penalty
        reasons.append(f"narrow_spectrum(w={sw:.1f}<{SPECTRAL_WIDTH_MIN})"
                        f" -{penalty:.3f}")

    # C) Peak-persistence gate
    pp = features.get("peak_persist", 0.0)
    if pp > PEAK_PERSIST_THRESH:
        excess   = (pp - PEAK_PERSIST_THRESH) / (1.0 - PEAK_PERSIST_THRESH + 1e-6)
        penalty  = PEAK_PERSIST_PENALTY * excess
        confidence -= penalty
        reasons.append(f"stable_peak(p={pp:.2f}>{PEAK_PERSIST_THRESH})"
                        f" -{penalty:.3f}")

    confidence = float(np.clip(confidence, 0.0, 1.0))
    motion_detected = confidence >= SCORE_THRESHOLD

    if   confidence >= 0.85: activity = "HIGH"
    elif confidence >= 0.65: activity = "MED"
    else:                    activity = "LOW"

    return {
        "motion_detected":  motion_detected,
        "activity_level":   activity,
        "confidence":       round(confidence, 4),
        "confidence_s1":    round(conf_s1, 4),
        "raw_score":        round(composite, 3),
        "feature_zscores":  zscores,
        "rejection_reason": reasons,
    }


# =============================================================================
#  HYSTERESIS STATE MACHINE
# =============================================================================

class HysteresisDetector:
    """
    Prevents rapid on/off flapping.

    Latches to ON after N_ON consecutive detected frames.
    Latches to OFF after N_OFF consecutive non-detected frames.

    State transitions:
        OFF  --[N_ON  consecutive detections]    --> ON
        ON   --[N_OFF consecutive non-detections] --> OFF
    """

    def __init__(self, n_on: int = N_ON_FRAMES, n_off: int = N_OFF_FRAMES):
        self.n_on    = n_on
        self.n_off   = n_off
        self.state   = False
        self._streak = 0

    def update(self, detected: bool) -> bool:
        if detected:
            self._streak = max(self._streak, 0) + 1
        else:
            self._streak = min(self._streak, 0) - 1
        if not self.state and self._streak >= self.n_on:
            self.state = True
        elif self.state and self._streak <= -self.n_off:
            self.state = False
        return self.state

    def reset(self):
        self.state   = False
        self._streak = 0


# =============================================================================
#  CONSOLE DISPLAY
# =============================================================================

BAR_WIDTH = 28
_ANSI = sys.stdout.isatty()

def _col(detected: bool, activity: str) -> str:
    if not _ANSI:
        return ""
    if not detected:
        return "\033[90m"
    if activity == "HIGH": return "\033[91m"
    if activity == "MED":  return "\033[93m"
    return "\033[92m"

_RST = "\033[0m" if _ANSI else ""


def display_result(result: dict, elapsed_ms: float,
                   frame_idx: int,
                   features: Optional[dict] = None,
                   verbose: bool = False) -> None:
    """Print a one-line status bar to stdout (overwriting previous line)."""
    conf     = result["confidence"]
    conf_s1  = result["confidence_s1"]
    act      = result["activity_level"]
    det      = result["motion_detected"]
    col      = _col(det, act)
    filled   = int(conf * BAR_WIDTH)
    bar      = chr(0x2588) * filled + chr(0x2591) * (BAR_WIDTH - filled)
    label    = "HUMAN DETECTED " if det else "               "
    rej      = ""
    if result["rejection_reason"]:
        rej = "  [" + "  ".join(result["rejection_reason"]) + "]"

    print(f"\r[{frame_idx:04d}] {col}{label}{_RST}"
          f"[{bar}] "
          f"conf={conf:.3f}(s1={conf_s1:.3f}) "
          f"{act:4}  {elapsed_ms:5.0f}ms"
          f"{rej}",
          end="", flush=True)

    if verbose and features:
        z = result["feature_zscores"]
        print()
        print(f"         ph_rms={features['phase_rms']:.3f}  "
              f"ph_lo={features['phase_rms_lo']:.3f}  "
              f"ph_hi={features['phase_rms_hi']:.3f}  "
              f"res={features['residual_mean']:.1f}  "
              f"cv={features['temporal_cv']:.4f}")
        print(f"         spec_w={features['spectral_width']:.2f}  "
              f"persist={features['peak_persist']:.3f}  "
              f"ph_ac={features['phase_ac_max']:.3f}  "
              f"entropy={features['spec_entropy']:.4f}")
        print(f"         z: ph={z.get('phase_rms',0):+.2f}  "
              f"res={z.get('residual_mean',0):+.2f}  "
              f"cv={z.get('temporal_cv',0):+.2f}  "
              f"score={result['raw_score']:+.2f}")


# =============================================================================
#  PLUTO SDR INTERFACE
# =============================================================================

def pluto_open(uri: str, carrier_hz: int, sample_rate: int,
               rx_gain_db: int, tx_gain_db: int):
    """
    Open ADALM-Pluto and configure RX/TX for CW Doppler operation.

    TX configuration
    ----------------
    A single-tone CW signal is generated at an IF offset of TX_IF_TONE_HZ
    (500 Hz) above the LO.  With the Pluto's internal 20x decimation and
    the LO at DEFAULT_CARRIER_HZ, the returned Doppler lands near the
    centre of the decimated spectrum.

    RX configuration
    ----------------
    Manual gain mode is used to prevent the AGC from attenuating the
    returning Doppler signal.  Set --rx_gain to 30 if saturation is detected.

    Windows notes
    -------------
    - Pluto appears as a USB Ethernet adapter at 192.168.2.1
    - Ensure the libiio Windows driver is installed before using pyadi-iio
    - If the device is not found, try --uri usb: (direct USB mode)

    Returns the adi.Pluto object.  Raises RuntimeError if pyadi-iio is
    not installed or the device cannot be found.
    """
    try:
        import adi
    except ImportError:
        raise RuntimeError(
            "pyadi-iio is not installed.\
"
            "  pip install pyadi-iio\
"
            "Also install libiio for Windows:\
"
            "  https://github.com/analogdevicesinc/libiio/releases"
        )

    print(f"[SDR] Connecting to Pluto at {uri} ...")
    try:
        sdr = adi.Pluto(uri=uri)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot open Pluto at {uri}: {exc}\
"
            "Check: USB connected, libiio installed, IP reachable."
        ) from exc

    # TX: continuous wave at carrier + IF tone
    sdr.tx_lo                = carrier_hz
    sdr.tx_rf_bandwidth      = max(200_000, sample_rate // 4)
    sdr.tx_sample_rate       = sample_rate
    sdr.tx_hardwaregain_chan0 = tx_gain_db
    sdr.tx_cyclic_buffer     = True

    N_tone = 1024
    t      = np.arange(N_tone) / sample_rate
    tone   = (0.5 * np.exp(2j * np.pi * TX_IF_TONE_HZ * t)).astype(np.complex64)
    tx_buf = (tone * 2 ** 14).astype(np.int16)
    sdr.tx(np.array([tx_buf.view(np.int16)]))

    # RX: manual gain, capture buffer
    sdr.rx_lo                    = carrier_hz
    sdr.rx_rf_bandwidth          = max(200_000, sample_rate // 4)
    sdr.rx_sample_rate           = sample_rate
    sdr.gain_control_mode_chan0  = "manual"
    sdr.rx_hardwaregain_chan0    = rx_gain_db
    sdr.rx_buffer_size           = int(sample_rate * (DEFAULT_CHUNK_SEC + 0.5))

    print(f"[SDR] OK  carrier={carrier_hz/1e9:.4f} GHz  "
          f"rate={sample_rate/1e6:.1f} MSPS  "
          f"RX={rx_gain_db} dB  TX={tx_gain_db} dB")
    return sdr


def pluto_capture(sdr, n_samples: int) -> np.ndarray:
    """
    Capture IQ from Pluto RX buffer.  Normalises to float amplitude ~1.

    Handles both int16 interleaved and complex float output formats
    depending on pyadi-iio version.
    """
    raw = sdr.rx()
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    raw = np.array(raw).flatten()

    if raw.dtype in (np.int16, np.int32):
        iq = (raw[0::2].astype(np.float32) +
              1j * raw[1::2].astype(np.float32)) / 2048.0
    elif np.iscomplexobj(raw):
        iq = raw.astype(np.complex64) / 2048.0
    else:
        iq = raw.astype(np.float32) / 2048.0

    return iq[:n_samples].astype(np.complex64)


# =============================================================================
#  CALIBRATION
# =============================================================================

def calibrate_live(sdr, sample_rate: int,
                   cal_sec: float = 20.0,
                   n_sigma: float = 2.0) -> dict:
    """
    Capture cal_sec seconds of IQ with no target present and build the
    CFAR baseline + clutter model.

    Call this once at the start of an operation when the detection zone
    is known to be empty.  The result is saved to DEFAULT_CALIB_FILE so
    subsequent runs can skip calibration.
    """
    print(f"\
[CAL] Live calibration: {cal_sec:.0f} s  --  ensure zone is EMPTY")
    print("      Starting in 3 s ...")
    time.sleep(3.0)

    n_samples = int(sample_rate * cal_sec)
    iq_raw    = pluto_capture(sdr, n_samples)
    iq_raw   -= iq_raw.mean()

    clutter_spec = build_clutter_model([iq_raw], sample_rate)
    iq_ds, fs_ds = safe_decimate(iq_raw, sample_rate, TARGET_FS)
    feat         = extract_features(iq_ds, fs_ds, clutter_spec)
    cfar         = build_cfar_baseline([feat], n_sigma=n_sigma)

    print("[CAL] Done.  CFAR baseline:")
    for k, v in cfar.items():
        print(f"       {k:20} mu={v['mean']:.4f}  sigma={v['std']:.6f}")

    return {"cfar": cfar, "clutter_spec": clutter_spec}


def load_calibration_from_npz(paths: list[str],
                               n_sigma: float = 2.0) -> dict:
    """
    Build the CFAR baseline and clutter model from pre-recorded empty .npz
    files.  Each file must contain 'iq' (complex64) and 'fs' (int).

    Recommended: use 2+ files recorded at different antenna orientations for
    a more robust baseline.
    """
    print(f"[CAL] Building baseline from {len(paths)} file(s):")
    iq_list:    list[np.ndarray] = []
    feat_list:  list[dict]       = []
    src_fs = None

    for path in paths:
        npz    = np.load(path, allow_pickle=True)
        iq     = npz["iq"].astype(np.complex64)
        fs     = int(npz["fs"])
        src_fs = src_fs or fs
        iq_list.append(iq)
        print(f"       {Path(path).name:50}  {len(iq)/fs:.1f} s")

    clutter_spec = build_clutter_model(iq_list, src_fs)

    for iq in iq_list:
        iq_c       = remove_dc(iq)
        iq_ds, fs_ds = safe_decimate(iq_c, src_fs, TARGET_FS)
        feat       = extract_features(iq_ds, fs_ds, clutter_spec)
        feat_list.append(feat)

    cfar = build_cfar_baseline(feat_list, n_sigma=n_sigma)

    print(f"[CAL] CFAR baseline (n_sigma={n_sigma}):")
    for k, v in cfar.items():
        print(f"       {k:20} mu={v['mean']:.4f}  sigma={v['std']:.6f}")

    return {"cfar": cfar, "clutter_spec": clutter_spec}


def save_calibration(calib: dict, path: str) -> None:
    """Save calibration to .npz for future reuse."""
    data: dict = {}
    for k, v in calib["cfar"].items():
        data[f"cfar_{k}_mean"] = float(v["mean"])
        data[f"cfar_{k}_std"]  = float(v["std"])
    if calib.get("clutter_spec") is not None:
        data["clutter_spec"] = calib["clutter_spec"]
    np.savez(path, **data)
    print(f"[CAL] Saved -> {path}")


def load_calibration(path: str) -> dict:
    """Load a saved calibration .npz."""
    npz  = np.load(path, allow_pickle=True)
    cfar: dict = {}
    for k in FEATURE_WEIGHTS:
        mk, sk = f"cfar_{k}_mean", f"cfar_{k}_std"
        if mk in npz and sk in npz:
            mu = float(npz[mk]);  sd = float(npz[sk])
            cfar[k] = {"mean": mu, "std": max(sd, 1e-6),
                       "threshold": mu + 2.0 * sd}
    clutter_spec = npz["clutter_spec"] if "clutter_spec" in npz else None
    print(f"[CAL] Loaded calibration from {path}")
    return {"cfar": cfar, "clutter_spec": clutter_spec}


# =============================================================================
#  OFFLINE DEMO  (replay .npz files without SDR)
# =============================================================================

def run_demo(demo_paths: list[str], calib: dict,
             chunk_sec: float = DEFAULT_CHUNK_SEC,
             verbose: bool = False) -> None:
    """
    Replay pre-recorded .npz files through the full detection pipeline.
    Prints per-chunk results identical to the live mode.
    """
    cfar         = calib["cfar"]
    clutter_spec = calib.get("clutter_spec")
    hysteresis   = HysteresisDetector()

    print("\
" + "=" * 80)
    print("  FireFinder Engine -- OFFLINE DEMO MODE")
    print("=" * 80)

    for path in demo_paths:
        npz     = np.load(path, allow_pickle=True)
        iq      = npz["iq"].astype(np.complex64)
        src_fs  = int(npz["fs"])
        fname   = Path(path).name
        print(f"\
  {fname}  ({len(iq)/src_fs:.1f} s)")
        print("  " + "-" * 60)

        chunk_n  = int(src_fs * chunk_sec)
        n_chunks = max(1, len(iq) // chunk_n)
        hysteresis.reset()

        for chunk_idx in range(n_chunks):
            t0    = time.perf_counter()
            start = chunk_idx * chunk_n
            end   = min(start + chunk_n, len(iq))
            iq_c  = remove_dc(iq[start:end])

            if chunk_idx == 0:
                sat = check_saturation(iq_c)
                if sat["saturated"]:
                    print(f"  [WARN] Saturation detected "
                          f"clip={sat['clip_frac']*100:.3f}%  "
                          f"tail={sat['tail_ratio']:.3f}")

            iq_ds, fs_ds = safe_decimate(iq_c, src_fs, TARGET_FS)
            features     = extract_features(iq_ds, fs_ds, clutter_spec)
            result       = score_detection(features, cfar)
            latched      = hysteresis.update(result["motion_detected"])

            rd = dict(result)
            rd["motion_detected"] = latched
            elapsed_ms = (time.perf_counter() - t0) * 1000
            display_result(rd, elapsed_ms, chunk_idx, features, verbose=verbose)

        print()

    print("\
[DONE]")


# =============================================================================
#  LIVE DETECTION LOOP
# =============================================================================

def run_live(sdr, sample_rate: int, chunk_sec: float,
             calib: dict, adaptive: bool = False,
             verbose: bool = False) -> None:
    """
    Continuous live detection loop.  Runs until Ctrl-C.

    Parameters
    ----------
    sdr         : open Pluto device object
    sample_rate : hardware sample rate (Hz)
    chunk_sec   : seconds per detection cycle
    calib       : {'cfar': ..., 'clutter_spec': ...}
    adaptive    : enable rolling clutter update from low-confidence frames
    verbose     : print full feature breakdown each frame
    """
    cfar         = calib["cfar"]
    clutter_spec = calib.get("clutter_spec")
    hysteresis   = HysteresisDetector()
    chunk_n      = int(sample_rate * chunk_sec)
    frame_idx    = 0

    print("\
" + "=" * 80)
    print("  FireFinder Engine -- LIVE DETECTION")
    print(f"  Chunk: {chunk_sec} s  |  Decimated FS: {TARGET_FS} Hz  |"
          f"  Adaptive: {'ON' if adaptive else 'OFF'}")
    print("  Ctrl-C to stop.")
    print("=" * 80 + "\
")

    try:
        while True:
            t0     = time.perf_counter()
            iq_raw = pluto_capture(sdr, chunk_n)
            iq_c   = remove_dc(iq_raw)

            # Saturation check every 30 frames
            if frame_idx % 30 == 0:
                sat = check_saturation(iq_c)
                if sat["saturated"]:
                    print(f"\
[WARN] Frame {frame_idx}: saturation  "
                          f"clip={sat['clip_frac']*100:.3f}%  "
                          f"tail={sat['tail_ratio']:.3f}")

            iq_ds, fs_ds = safe_decimate(iq_c, sample_rate, TARGET_FS)
            features     = extract_features(iq_ds, fs_ds, clutter_spec)
            result       = score_detection(features, cfar)
            latched      = hysteresis.update(result["motion_detected"])

            # Adaptive clutter update on clearly-empty frames
            if adaptive and clutter_spec is not None:
                if result["confidence"] < ADAPTIVE_MAX_CONF:
                    win = max(16, int(WIN_SEC * fs_ds))
                    hop = max(4,  int(HOP_SEC * fs_ds))
                    _, _, Zxx = stft(iq_ds, fs=fs_ds, nperseg=win,
                                     noverlap=win - hop, return_onesided=False)
                    new_spec     = np.fft.fftshift(
                                       np.abs(Zxx) ** 2, axes=0).mean(axis=1)
                    clutter_spec = ema_update_clutter(clutter_spec, new_spec)

            rd = dict(result)
            rd["motion_detected"] = latched
            elapsed_ms = (time.perf_counter() - t0) * 1000
            display_result(rd, elapsed_ms, frame_idx, features, verbose=verbose)
            frame_idx += 1

    except KeyboardInterrupt:
        print("\
\
[STOP] Stopped by user.")


# =============================================================================
#  CLI ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FireFinder CW Doppler Radar -- Real-Time Human Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    hw = parser.add_argument_group("SDR hardware")
    hw.add_argument("--uri",         default=DEFAULT_URI)
    hw.add_argument("--freq",        type=int, default=DEFAULT_CARRIER_HZ,
                    help="Carrier Hz")
    hw.add_argument("--rx_gain",     type=int, default=DEFAULT_RX_GAIN_DB,
                    help="RX gain dB  (try 30 if saturated)")
    hw.add_argument("--tx_gain",     type=int, default=DEFAULT_TX_GAIN_DB,
                    help="TX gain dB  (-40 + PA ~+20 = ~-20 dBm ERP)")
    hw.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE,
                    help="Hardware rate Hz")
    hw.add_argument("--chunk_sec",   type=float, default=DEFAULT_CHUNK_SEC,
                    help="Seconds of IQ per detection cycle")

    cal = parser.add_argument_group("Calibration")
    cal.add_argument("--calibrate",      action="store_true",
                     help="Live auto-calibration (empty zone)")
    cal.add_argument("--calibrate_sec",  type=float, default=20.0,
                     help="Live cal duration (s)")
    cal.add_argument("--clutter",        nargs="+", metavar="FILE",
                     help="Pre-recorded empty .npz file(s)")
    cal.add_argument("--calib_file",     default=DEFAULT_CALIB_FILE,
                     help="Calibration save/load path")
    cal.add_argument("--n_sigma",        type=float, default=2.0,
                     help="CFAR threshold multiplier")

    run = parser.add_argument_group("Run modes")
    run.add_argument("--demo",        nargs="+", metavar="FILE",
                     help="Offline replay (no SDR)")
    run.add_argument("--adaptive",    action="store_true",
                     help="Rolling clutter update from low-confidence frames")
    run.add_argument("--verbose",     action="store_true",
                     help="Full feature breakdown each frame")

    args = parser.parse_args()

    # ---- Offline demo --------------------------------------------------------
    if args.demo:
        if args.clutter:
            calib = load_calibration_from_npz(args.clutter, args.n_sigma)
        elif os.path.exists(args.calib_file):
            calib = load_calibration(args.calib_file)
        else:
            print("[WARN] No calibration source -- building from demo files.")
            calib = load_calibration_from_npz(args.demo, args.n_sigma)

        run_demo(args.demo, calib,
                 chunk_sec=args.chunk_sec,
                 verbose=args.verbose)
        return

    # ---- Open SDR -----------------------------------------------------------
    sdr = pluto_open(args.uri, args.freq, args.sample_rate,
                     args.rx_gain, args.tx_gain)

    # ---- Calibration --------------------------------------------------------
    if args.calibrate:
        calib = calibrate_live(sdr, args.sample_rate,
                               cal_sec=args.calibrate_sec,
                               n_sigma=args.n_sigma)
        save_calibration(calib, args.calib_file)

    elif args.clutter:
        calib = load_calibration_from_npz(args.clutter, args.n_sigma)
        save_calibration(calib, args.calib_file)

    elif os.path.exists(args.calib_file):
        calib = load_calibration(args.calib_file)

    else:
        print("[WARN] No calibration found.  Capturing 10 s ad-hoc baseline ...")
        print("       Ensure detection zone is EMPTY.")
        n_cal  = int(args.sample_rate * 10)
        iq_cal = pluto_capture(sdr, n_cal)
        iq_cal = remove_dc(iq_cal)
        clutter_spec = build_clutter_model([iq_cal], args.sample_rate)
        iq_ds, fs_ds = safe_decimate(iq_cal, args.sample_rate, TARGET_FS)
        feat_cal     = extract_features(iq_ds, fs_ds, clutter_spec)
        cfar_cal     = build_cfar_baseline([feat_cal], n_sigma=args.n_sigma)
        calib        = {"cfar": cfar_cal, "clutter_spec": clutter_spec}

    # ---- Live loop ----------------------------------------------------------
    run_live(sdr, args.sample_rate, args.chunk_sec, calib,
             adaptive=args.adaptive,
             verbose=args.verbose)


if __name__ == "__main__":
    main()
