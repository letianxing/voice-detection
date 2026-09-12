from __future__ import annotations

import numpy as np

from .types import AudioFeatures


def extract_features(mono: np.ndarray, sample_rate_hz: int) -> AudioFeatures:
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return AudioFeatures(zcr=0.0, rms=0.0)

    rms = float(np.sqrt(np.mean(np.square(mono))))
    signs = np.signbit(mono)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if mono.size > 1 else 0.0
    pitch = estimate_pitch(mono, sample_rate_hz)
    hnr = estimate_hnr_proxy(mono)
    return AudioFeatures(zcr=zcr, rms=rms, pitch=pitch, hnr=hnr)


def estimate_pitch(mono: np.ndarray, sample_rate_hz: int) -> float:
    centered = mono - float(np.mean(mono))
    if centered.size < 3 or float(np.max(np.abs(centered))) < 1e-5:
        return 0.0
    corr = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    min_lag = max(1, int(sample_rate_hz / 500.0))
    max_lag = min(corr.size - 1, int(sample_rate_hz / 70.0))
    if max_lag <= min_lag:
        return 0.0
    lag = int(np.argmax(corr[min_lag:max_lag]) + min_lag)
    if lag <= 0:
        return 0.0
    pitch_hz = sample_rate_hz / lag
    return float(max(0.0, min(1.0, pitch_hz / 500.0)))


def estimate_hnr_proxy(mono: np.ndarray) -> float:
    energy = float(np.mean(np.square(mono)))
    if energy < 1e-9:
        return 0.0
    diff_energy = float(np.mean(np.square(np.diff(mono)))) if mono.size > 1 else 0.0
    return float(max(0.0, min(1.0, 1.0 - (diff_energy / (energy * 8.0 + 1e-9)))))

