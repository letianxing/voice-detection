from __future__ import annotations

import numpy as np


class AdaptiveNoiseSuppressor:
    def __init__(self, noise_alpha: float = 0.98, max_reduction: float = 0.35):
        self.noise_alpha = noise_alpha
        self.max_reduction = max_reduction
        self.noise_rms = 0.003

    def process(self, mono: np.ndarray, speech_active_hint: bool) -> np.ndarray:
        data = np.asarray(mono, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return data
        rms = float(np.sqrt(np.mean(np.square(data))) + 1e-9)
        if not speech_active_hint:
            self.noise_rms = (self.noise_alpha * self.noise_rms) + ((1.0 - self.noise_alpha) * rms)
        snr = rms / max(self.noise_rms, 1e-6)
        gain = max(self.max_reduction, min(1.0, 0.55 + (0.45 * ((snr - 1.0) / 6.0))))
        return (data * gain).astype(np.float32)


def remove_dc(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    return data - np.mean(data, axis=0, keepdims=True)


def estimate_clarity_from_snr(snr_db: float) -> float:
    return float(max(0.0, min(1.0, snr_db / 30.0)))


def estimate_overlap_probability(samples: np.ndarray) -> float:
    """Cheap overlap proxy based on channel energy spread and waveform shape."""

    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1 or data.shape[1] < 2:
        return 0.0
    channel_rms = np.sqrt(np.mean(np.square(data), axis=0) + 1e-9)
    spread = float(np.std(channel_rms) / (np.mean(channel_rms) + 1e-9))
    mono = np.mean(data, axis=1)
    centered = mono - float(np.mean(mono))
    var = float(np.mean(np.square(centered)) + 1e-9)
    kurtosis = float(np.mean(centered**4) / (var * var))
    diffuse = max(0.0, min(1.0, 1.0 - spread))
    flatness = max(0.0, min(1.0, (kurtosis - 1.5) / 6.0))
    return float(max(0.0, min(1.0, (0.65 * diffuse) + (0.35 * flatness))))


def estimate_self_echo_probability(mono: np.ndarray, playback_reference: np.ndarray | None) -> float:
    if playback_reference is None:
        return 0.0
    source = np.asarray(mono, dtype=np.float32).reshape(-1)
    reference = np.asarray(playback_reference, dtype=np.float32).reshape(-1)
    n = min(source.size, reference.size)
    if n < 32:
        return 0.0
    source = source[:n] - float(np.mean(source[:n]))
    reference = reference[:n] - float(np.mean(reference[:n]))
    denom = float(np.linalg.norm(source) * np.linalg.norm(reference) + 1e-9)
    corr = float(abs(np.dot(source, reference)) / denom)
    return max(0.0, min(1.0, corr))
