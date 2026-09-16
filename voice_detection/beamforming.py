from __future__ import annotations

from math import cos, radians, sin
from typing import Iterable

import numpy as np

from .doa import SPEED_OF_SOUND_M_S


def mixdown(samples: np.ndarray, channel_indices: Iterable[int] | None = None) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        return data
    if channel_indices is not None:
        indices = tuple(int(index) for index in channel_indices)
        if indices:
            data = data[:, indices]
    live = np.any(np.abs(data) > 1e-9, axis=0)
    # Digital-zero channels must not attenuate a usable microphone by 8x.
    return np.mean(data[:, live], axis=1) if np.any(live) else np.zeros(data.shape[0], dtype=np.float32)


def delay_and_sum(
    samples: np.ndarray,
    sample_rate_hz: int,
    mic_positions_m: Iterable[Iterable[float]],
    channel_indices: Iterable[int],
    azimuth_deg: float,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("samples must be [frames, channels]")
    indices = tuple(int(index) for index in channel_indices)
    if len(indices) < 2:
        return mixdown(data, indices)
    positions = np.asarray(list(mic_positions_m), dtype=np.float32)
    direction = np.array([cos(radians(azimuth_deg)), sin(radians(azimuth_deg)), 0.0], dtype=np.float32)
    selected = []
    delays = []
    for index in indices:
        arrival_time = -float(np.dot(positions[index], direction)) / speed_of_sound_m_s
        delays.append(arrival_time)
    min_delay = min(delays)
    normalized_delays = [delay - min_delay for delay in delays]
    for index, delay_s in zip(indices, normalized_delays):
        shift = int(round(delay_s * sample_rate_hz))
        channel = data[:, index]
        if shift <= 0:
            aligned = channel
        else:
            aligned = np.pad(channel[shift:], (0, shift))
        selected.append(aligned)
    return np.mean(np.stack(selected, axis=1), axis=1).astype(np.float32)

