from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from typing import Iterable

import numpy as np


SPEED_OF_SOUND_M_S = 343.0


@dataclass(frozen=True)
class DoaEstimate:
    azimuth_deg: float | None
    confidence: float
    pair_count: int


def gcc_phat(sig: np.ndarray, refsig: np.ndarray, sample_rate_hz: int, max_tau: float | None = None) -> float:
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    refsig = np.asarray(refsig, dtype=np.float32).reshape(-1)
    n = sig.size + refsig.size
    if n <= 2:
        return 0.0
    sig_fft = np.fft.rfft(sig, n=n)
    ref_fft = np.fft.rfft(refsig, n=n)
    cross = sig_fft * np.conj(ref_fft)
    cross /= np.maximum(np.abs(cross), 1e-12)
    cc = np.fft.irfft(cross, n=n)
    max_shift = n // 2
    if max_tau is not None:
        max_shift = min(max_shift, max(1, int(sample_rate_hz * max_tau)))
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc)) - max_shift)
    return float(shift / float(sample_rate_hz))


def estimate_azimuth_gcc(
    samples: np.ndarray,
    sample_rate_hz: int,
    mic_positions_m: Iterable[Iterable[float]],
    channel_indices: Iterable[int],
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
    grid_step_deg: float = 2.0,
) -> DoaEstimate:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("samples must be [frames, channels]")
    indices = tuple(int(index) for index in channel_indices)
    positions = np.asarray(list(mic_positions_m), dtype=np.float32)
    if len(indices) < 2 or positions.shape[0] <= max(indices):
        return DoaEstimate(azimuth_deg=None, confidence=0.0, pair_count=0)

    if max(indices) >= data.shape[1]:
        return DoaEstimate(None, 0.0, 0)
    rms = np.sqrt(np.mean((data[:, indices]-data[:, indices].mean(axis=0))**2, axis=0))
    active = rms > max(1e-6, float(rms.max())*.01)
    # Missing raw channels cannot produce a physically meaningful bearing.
    if not np.all(active):
        return DoaEstimate(None, 0.0, 0)
    measured: list[tuple[int, int, float]] = []
    for pos, left in enumerate(indices):
        for right in indices[pos + 1 :]:
            baseline = positions[left] - positions[right]
            max_tau = float(np.linalg.norm(baseline) / speed_of_sound_m_s)
            tau = gcc_phat(data[:, left], data[:, right], sample_rate_hz, max_tau=max_tau)
            measured.append((left, right, tau))

    if not measured:
        return DoaEstimate(azimuth_deg=None, confidence=0.0, pair_count=0)

    best_azimuth = 0.0
    best_error = float("inf")
    for azimuth in np.arange(-180.0, 180.0, grid_step_deg):
        direction = np.array([cos(radians(azimuth)), sin(radians(azimuth)), 0.0], dtype=np.float32)
        error = 0.0
        for left, right, tau in measured:
            expected = -float(np.dot(positions[left] - positions[right], direction)) / speed_of_sound_m_s
            error += abs(tau - expected)
        error /= len(measured)
        if error < best_error:
            best_error = error
            best_azimuth = float(azimuth)

    max_delay = max(
        float(np.linalg.norm(positions[left] - positions[right]) / speed_of_sound_m_s)
        for left, right, _ in measured
    )
    confidence = max(0.0, min(1.0, 1.0 - (best_error / max(max_delay, 1e-6))))
    return DoaEstimate(
        azimuth_deg=normalize_angle_deg(best_azimuth),
        confidence=float(confidence),
        pair_count=len(measured),
    )


def normalize_angle_deg(angle: float) -> float:
    normalized = (float(angle) + 180.0) % 360.0 - 180.0
    if normalized == -180.0:
        return 180.0
    return normalized

