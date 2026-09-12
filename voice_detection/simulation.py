from __future__ import annotations

from math import cos, radians, sin

import numpy as np

from .doa import SPEED_OF_SOUND_M_S
from .types import AudioFrame, MicrophoneProfile


def synthetic_array_frame(
    profile: MicrophoneProfile,
    azimuth_deg: float = 30.0,
    duration_ms: int = 120,
    frequency_hz: float = 220.0,
    amplitude: float = 0.3,
    seed: int = 7,
    stamp_ms: int = 1000,
) -> AudioFrame:
    sample_rate = profile.sample_rate_hz
    samples_count = int(sample_rate * duration_ms / 1000.0)
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.4, samples_count).astype(np.float32)
    t = np.arange(samples_count, dtype=np.float32) / float(sample_rate)
    base += np.sin(2.0 * np.pi * frequency_hz * t).astype(np.float32)
    base *= amplitude / max(float(np.max(np.abs(base))), 1e-6)

    channels = max(profile.input_channels, 1)
    data = np.zeros((samples_count, channels), dtype=np.float32)
    direction = np.array([cos(radians(azimuth_deg)), sin(radians(azimuth_deg)), 0.0], dtype=np.float32)
    positions = np.asarray(profile.mic_positions_m, dtype=np.float32)
    if positions.shape[0] < channels:
        positions = np.pad(positions, ((0, channels - positions.shape[0]), (0, 0)))
    arrivals = [-float(np.dot(positions[index], direction)) / SPEED_OF_SOUND_M_S for index in range(channels)]
    min_arrival = min(arrivals)
    for index, arrival in enumerate(arrivals):
        shift = int(round((arrival - min_arrival) * sample_rate))
        if shift <= 0:
            data[:, index] = base
        elif shift < samples_count:
            data[shift:, index] = base[:-shift]
    data += rng.normal(0.0, 0.005, data.shape).astype(np.float32)
    return AudioFrame(samples=data, sample_rate_hz=sample_rate, stamp_ms=stamp_ms)

