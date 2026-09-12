from __future__ import annotations

import numpy as np

from .audio_frame_codec import float32_to_pcm16


class DelayedCrossfadeJoiner:
    """Joins streaming decoder chunks while delaying the mutable overlap tail."""

    def __init__(self, overlap_samples: int):
        self.overlap_samples = max(0, int(overlap_samples))
        self._tail = np.zeros(0, dtype=np.float32)

    def append(self, chunk: np.ndarray) -> np.ndarray:
        data = _mono(chunk)
        if data.size == 0:
            return data
        overlap = self.overlap_samples
        if overlap == 0:
            return data
        if self._tail.size == 0:
            if data.size <= overlap:
                self._tail = data.copy()
                return np.zeros(0, dtype=np.float32)
            self._tail = data[-overlap:].copy()
            return data[:-overlap].copy()

        count = min(overlap, self._tail.size, data.size)
        fade_in = _fade(count)
        mixed = (self._tail[-count:] * (1.0 - fade_in)) + (data[:count] * fade_in)
        prefix = self._tail[:-count]
        if data.size > overlap:
            middle = data[count:-overlap]
            self._tail = data[-overlap:].copy()
        else:
            middle = np.zeros(0, dtype=np.float32)
            self._tail = data[count:].copy()
        return np.concatenate((prefix, mixed, middle)).astype(np.float32)

    def flush(self) -> np.ndarray:
        result = self._tail.copy()
        self._tail = np.zeros(0, dtype=np.float32)
        return result


class PcmPacketizer:
    """Crossfades first, then emits fixed-size PCM16 playback/AEC packets."""

    def __init__(self, packet_samples: int, overlap_samples: int = 320):
        if packet_samples <= 0:
            raise ValueError("packet_samples must be greater than zero")
        self.packet_samples = int(packet_samples)
        self._joiner = DelayedCrossfadeJoiner(overlap_samples)
        self._buffer = np.zeros(0, dtype=np.int16)

    def push(self, decoded_chunk: np.ndarray) -> list[np.ndarray]:
        self._append(self._joiner.append(decoded_chunk))
        return self._take(force=False)

    def flush(self) -> list[np.ndarray]:
        self._append(self._joiner.flush())
        return self._take(force=True)

    def _append(self, samples: np.ndarray) -> None:
        pcm = float32_to_pcm16(samples)
        if pcm.size:
            self._buffer = np.concatenate((self._buffer, pcm))

    def _take(self, force: bool) -> list[np.ndarray]:
        packets = []
        while self._buffer.size >= self.packet_samples:
            packets.append(self._buffer[: self.packet_samples].copy())
            self._buffer = self._buffer[self.packet_samples :]
        if force and self._buffer.size:
            packet = np.zeros(self.packet_samples, dtype=np.int16)
            packet[: self._buffer.size] = self._buffer
            packets.append(packet)
            self._buffer = np.zeros(0, dtype=np.int16)
        return packets


def _mono(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim <= 1:
        return data.reshape(-1)
    return np.mean(data, axis=1).astype(np.float32)


def _fade(count: int) -> np.ndarray:
    if count <= 1:
        return np.ones(max(0, count), dtype=np.float32)
    phase = np.linspace(0.0, 1.0, count, dtype=np.float32)
    return 0.5 - (0.5 * np.cos(np.pi * phase))
