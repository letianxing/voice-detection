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
    """How much of this block is not one dominant voice.

    The previous version scored channel-energy uniformity and high kurtosis as
    evidence of overlap. Both are backwards on the hardware we run: on a 35 mm
    ring array one talker reaches all six microphones at nearly the same level,
    and voiced speech is strongly super-Gaussian while a mixture of voices tends
    towards Gaussian. Measured on synthetic blocks it returned 0.83 for one
    talker and 0.80 for two, and in the field it sat at 0.65 for a single person
    in a quiet room — which silently disabled everything gated on clean audio:
    the fusion's speech path, the orienting guard, and voice-pool learning.

    What is measured now is evidence *for* a single dominant source, and overlap
    is its complement:
      periodicity — one voice has one pitch, so the autocorrelation over 80-400
                    Hz peaks sharply; two voices blur it.
      coherence   — a single point source reaches opposite microphones as the
                    same waveform a few samples apart. Needs ≥2 channels.
      peakiness   — voiced speech is super-Gaussian, a mixture is closer to
                    Gaussian. The old code had this exactly inverted.
    Still a proxy, not a separation model, and not calibrated on real overlapping
    speech — but it now moves in the right direction, which it did not before.
    """
    data = np.asarray(samples, dtype=np.float32)
    mono = data if data.ndim == 1 else np.mean(data, axis=1)
    mono = mono - float(np.mean(mono))
    energy = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
    if mono.size < 256 or energy < 1e-5:
        return 0.0

    evidence, weights = [], []
    spectrum = np.correlate(mono, mono, mode="full")[mono.size - 1:]
    if spectrum[0] > 0:
        # 80-400 Hz at whatever rate this block is; 48 kHz gives lags 120-600.
        low, high = max(8, mono.size // 80), min(mono.size - 1, mono.size // 4)
        if high > low:
            periodicity = float(np.max(spectrum[low:high]) / spectrum[0])
            evidence.append(max(0.0, min(1.0, periodicity * 1.4)))
            weights.append(0.5)

    if data.ndim > 1 and data.shape[1] >= 2:
        left = data[:, 0] - float(np.mean(data[:, 0]))
        right = data[:, data.shape[1] // 2] - float(np.mean(data[:, data.shape[1] // 2]))
        norm = float(np.linalg.norm(left) * np.linalg.norm(right)) + 1e-9
        lags = range(-8, 9)
        coherence = max(float(np.dot(left, np.roll(right, lag))) / norm for lag in lags)
        evidence.append(max(0.0, min(1.0, coherence)))
        weights.append(0.3)

    variance = float(np.mean(np.square(mono)) + 1e-12)
    kurtosis = float(np.mean(mono ** 4) / (variance * variance))
    evidence.append(max(0.0, min(1.0, (kurtosis - 3.0) / 5.0)))
    weights.append(0.2)

    single_source = sum(value * weight for value, weight in zip(evidence, weights)) / sum(weights)
    return float(max(0.0, min(1.0, 1.0 - single_source)))


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
