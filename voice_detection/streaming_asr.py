from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import queue
import threading
import time
from typing import Callable, Deque

import numpy as np

from .asr import AsrAdapter, NullAsrAdapter
from .types import SpeechTranscript


@dataclass(frozen=True)
class Utterance:
    track_id: str
    samples: np.ndarray
    sample_rate_hz: int
    started_ms: int
    ended_ms: int


class UtteranceSegmenter:
    def __init__(
        self,
        pre_roll_ms: int = 180,
        endpoint_silence_ms: int = 480,
        min_speech_ms: int = 80,
        max_utterance_ms: int = 20_000,
    ):
        self.pre_roll_ms = max(0, int(pre_roll_ms))
        self.endpoint_silence_ms = max(40, int(endpoint_silence_ms))
        self.min_speech_ms = max(40, int(min_speech_ms))
        self.max_utterance_ms = max(self.min_speech_ms, int(max_utterance_ms))
        self._pre_roll: Deque[tuple[np.ndarray, int]] = deque()
        self._chunks: list[np.ndarray] = []
        self._track_id = ""
        self._started_ms = 0
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        stamp_ms: int,
        speech_active: bool,
        track_id: str,
    ) -> Utterance | None:
        data = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        duration_ms = 1000.0 * data.size / max(1, sample_rate_hz)
        if not self._chunks:
            self._append_pre_roll(data, stamp_ms, duration_ms)
            if not speech_active:
                return None
            self._chunks = [chunk for chunk, _stamp in self._pre_roll]
            self._started_ms = self._pre_roll[0][1] if self._pre_roll else stamp_ms
            self._track_id = track_id or "voice_mono"
            self._speech_ms = duration_ms
            self._silence_ms = 0.0
            self._pre_roll.clear()
            return None

        self._chunks.append(data)
        if speech_active:
            self._speech_ms += duration_ms
            self._silence_ms = 0.0
        else:
            self._silence_ms += duration_ms
        total_ms = 1000.0 * sum(chunk.size for chunk in self._chunks) / max(1, sample_rate_hz)
        if self._silence_ms < self.endpoint_silence_ms and total_ms < self.max_utterance_ms:
            return None
        return self._finish(sample_rate_hz, stamp_ms + int(duration_ms))

    def flush(self, sample_rate_hz: int, stamp_ms: int) -> Utterance | None:
        return self._finish(sample_rate_hz, stamp_ms) if self._chunks else None

    def _finish(self, sample_rate_hz: int, ended_ms: int) -> Utterance | None:
        samples = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
        utterance = None
        if self._speech_ms >= self.min_speech_ms:
            utterance = Utterance(
                track_id=self._track_id,
                samples=samples,
                sample_rate_hz=sample_rate_hz,
                started_ms=self._started_ms,
                ended_ms=ended_ms - int(self._silence_ms),
            )
        self._chunks = []
        self._track_id = ""
        self._started_ms = 0
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        return utterance

    def _append_pre_roll(self, data: np.ndarray, stamp_ms: int, duration_ms: float) -> None:
        self._pre_roll.append((data, stamp_ms))
        retained_ms = duration_ms * len(self._pre_roll)
        while self._pre_roll and retained_ms > self.pre_roll_ms + duration_ms:
            self._pre_roll.popleft()
            retained_ms -= duration_ms


class AsrWorker:
    def __init__(
        self,
        adapter: AsrAdapter | None,
        on_transcript: Callable[[SpeechTranscript], None],
        on_error: Callable[[str], None] | None = None,
    ):
        self.adapter = adapter or NullAsrAdapter()
        self.on_transcript = on_transcript
        self.on_error = on_error or (lambda _error: None)
        self.items: queue.Queue[Utterance] = queue.Queue(maxsize=4)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="asr-worker", daemon=True)
        self.thread.start()

    def submit(self, utterance: Utterance) -> bool:
        try:
            self.items.put_nowait(utterance)
            return True
        except queue.Full:
            self.on_error("asr_queue_full")
            return False

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                utterance = self.items.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                result = self.adapter.transcribe(
                    utterance.track_id,
                    utterance.samples,
                    utterance.sample_rate_hz,
                )
                if result is not None:
                    self.on_transcript(
                        replace(
                            result,
                            started_ms=utterance.started_ms,
                            ended_ms=utterance.ended_ms,
                            emitted_ms=int(time.time() * 1000),
                        )
                    )
            except Exception as exc:
                self.on_error(str(exc))
