from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import threading
import time
from typing import Deque

import numpy as np

from .audio_frame_codec import decode_audio_frame_payload
from .types import AudioFrame


class LinearEchoCanceller:
    """Low-cost aligned reference canceller; replaceable by WebRTC AEC3 in deployment."""

    def __init__(self, smoothing: float = 0.75, max_gain: float = 2.0):
        self.smoothing = float(np.clip(smoothing, 0.0, 0.999))
        self.max_gain = max(0.0, float(max_gain))
        self.echo_gain = 0.0

    def process(self, microphone: np.ndarray, playback_reference: np.ndarray | None) -> np.ndarray:
        source = np.asarray(microphone, dtype=np.float32).reshape(-1)
        if playback_reference is None:
            return source.copy()
        reference = np.asarray(playback_reference, dtype=np.float32).reshape(-1)
        count = min(source.size, reference.size)
        if count < 32:
            return source.copy()
        ref = reference[:count] - float(np.mean(reference[:count]))
        ref_energy = float(np.dot(ref, ref))
        if ref_energy < 1e-8:
            return source.copy()
        mic = source[:count] - float(np.mean(source[:count]))
        estimate = float(np.clip(np.dot(mic, ref) / ref_energy, -self.max_gain, self.max_gain))
        self.echo_gain = (self.smoothing * self.echo_gain) + ((1.0 - self.smoothing) * estimate)
        result = source.copy()
        result[:count] -= self.echo_gain * reference[:count]
        return np.clip(result, -1.0, 1.0).astype(np.float32)


class PlaybackReferenceBuffer:
    """Aligns timestamped TTS frames with microphone capture frames."""

    def __init__(self, retention_ms: int = 3000):
        self.retention_ms = max(100, int(retention_ms))
        self._frames: Deque[AudioFrame] = deque()
        self._lock = threading.Lock()

    def append(self, frame: AudioFrame) -> None:
        mono = np.mean(frame.samples, axis=1, keepdims=True).astype(np.float32)
        with self._lock:
            self._frames.append(AudioFrame(mono, frame.sample_rate_hz, frame.stamp_ms))
            self._discard_before(frame.stamp_ms - self.retention_ms)

    def read_for(self, capture: AudioFrame) -> np.ndarray:
        output = np.zeros(capture.samples.shape[0], dtype=np.float32)
        start_ms = capture.stamp_ms
        end_ms = start_ms + (1000.0 * output.size / capture.sample_rate_hz)
        with self._lock:
            self._discard_before(int(start_ms - self.retention_ms))
            for frame in self._frames:
                samples = _resample(np.mean(frame.samples, axis=1), frame.sample_rate_hz, capture.sample_rate_hz)
                frame_end_ms = frame.stamp_ms + (1000.0 * samples.size / capture.sample_rate_hz)
                overlap_start = max(float(start_ms), float(frame.stamp_ms))
                overlap_end = min(float(end_ms), frame_end_ms)
                if overlap_end <= overlap_start:
                    continue
                out_start = int(round((overlap_start - start_ms) * capture.sample_rate_hz / 1000.0))
                ref_start = int(round((overlap_start - frame.stamp_ms) * capture.sample_rate_hz / 1000.0))
                count = min(
                    output.size - out_start,
                    samples.size - ref_start,
                    int(round((overlap_end - overlap_start) * capture.sample_rate_hz / 1000.0)),
                )
                if count > 0:
                    output[out_start : out_start + count] = samples[ref_start : ref_start + count]
        return output

    def _discard_before(self, stamp_ms: int) -> None:
        while self._frames:
            frame = self._frames[0]
            duration_ms = 1000.0 * frame.samples.shape[0] / frame.sample_rate_hz
            if frame.stamp_ms + duration_ms >= stamp_ms:
                break
            self._frames.popleft()


class JsonlPlaybackReferenceSource:
    """Tails AudioFrame JSONL emitted by the ROS2 TTS bridge."""

    def __init__(
        self,
        path: str | Path,
        poll_interval_sec: float = 0.01,
        replay_existing: bool = False,
    ):
        self.path = Path(path).expanduser()
        self.offset = self.path.stat().st_size if self.path.exists() and not replay_existing else 0
        self.buffer = PlaybackReferenceBuffer()
        self._poll_interval_sec = max(0.002, float(poll_interval_sec))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="aec-reference-jsonl", daemon=True)
        self._thread.start()

    def read_for(self, capture: AudioFrame) -> np.ndarray:
        return self.buffer.read_for(capture)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def poll(self) -> int:
        if not self.path.exists():
            return 0
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
        loaded = 0
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            for line in handle:
                if line.strip():
                    self.buffer.append(decode_audio_frame_payload(json.loads(line)))
                    loaded += 1
            self.offset = handle.tell()
        return loaded

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            self._stop.wait(self._poll_interval_sec)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate or data.size == 0:
        return data
    output_size = max(1, int(round(data.size * target_rate / source_rate)))
    source_x = np.linspace(0.0, 1.0, data.size, endpoint=False)
    target_x = np.linspace(0.0, 1.0, output_size, endpoint=False)
    return np.interp(target_x, source_x, data).astype(np.float32)
