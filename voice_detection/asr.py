from __future__ import annotations

import json
import multiprocessing as mp
import queue
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .types import SpeechTranscript


@dataclass(frozen=True)
class AsrScenarioRecord:
    scenario_id: str
    utterance_expected: str
    environment: dict[str, object] = field(default_factory=dict)
    speaking_style: dict[str, object] = field(default_factory=dict)
    asr_result: SpeechTranscript | None = None
    diff_notes: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


class AsrAdapter:
    def transcribe(self, track_id: str, audio: np.ndarray, sample_rate_hz: int) -> SpeechTranscript | None:
        raise NotImplementedError


class NullAsrAdapter(AsrAdapter):
    def transcribe(self, track_id: str, audio: np.ndarray, sample_rate_hz: int) -> SpeechTranscript | None:
        return None


class VoskAsrAdapter(AsrAdapter):
    """Offline ASR adapter used for the Mac smoke-test path."""

    def __init__(self, model_path: str | Path, language: str = "zh-CN"):
        try:
            from vosk import Model, SetLogLevel
        except Exception as exc:
            raise RuntimeError(f"vosk is required for local ASR: {exc}") from exc
        SetLogLevel(-1)
        self.model = Model(str(Path(model_path).expanduser()))
        self.language = language

    def transcribe(self, track_id: str, audio: np.ndarray, sample_rate_hz: int) -> SpeechTranscript | None:
        from vosk import KaldiRecognizer

        mono = resample_mono(audio, sample_rate_hz, 16000)
        pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
        recognizer = KaldiRecognizer(self.model, 16000)
        recognizer.SetWords(True)
        recognizer.AcceptWaveform(pcm.tobytes())
        payload = json.loads(recognizer.FinalResult())
        text = str(payload.get("text") or "").strip()
        if not text:
            return None
        words = payload.get("result") or []
        confidence = sum(float(item.get("conf", 0.0)) for item in words) / len(words) if words else 0.0
        return SpeechTranscript(
            track_id=track_id,
            text=text.replace(" ", ""),
            is_final=True,
            language=self.language,
            confidence=confidence,
        )

    def create_stream(self, track_id: str, sample_rate_hz: int):
        return VoskStreamingSession(self.model, track_id, sample_rate_hz, self.language)


class SherpaZipformerAsrAdapter(AsrAdapter):
    """Official sherpa-onnx online Zipformer recognizer (Chinese, streaming)."""

    def __init__(self, model_dir: str | Path, language: str = "zh-CN"):
        root = Path(model_dir).expanduser()
        if not (root / "encoder.int8.onnx").is_file():
            raise RuntimeError(f"Sherpa model files not found: {root}")
        self.model_dir = str(root)
        self.language = language
        self.engine = None

    def create_stream(self, track_id: str, sample_rate_hz: int):
        self.warmup()
        return self.engine.session(track_id, self.language)

    def warmup(self):
        if self.engine is None:
            from .persistent_asr import SherpaEngine
            self.engine = SherpaEngine(self.model_dir)

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None

    def transcribe(self, track_id: str, audio: np.ndarray, sample_rate_hz: int) -> SpeechTranscript | None:
        session = self.create_stream(track_id, sample_rate_hz)
        session.accept(audio, sample_rate_hz, 0)
        result = session.finish(0)
        return result


class SherpaProcessStreamingSession:
    """Sherpa session in a child process to isolate native ONNX runtimes."""

    def __init__(self, model_dir: str, track_id: str, sample_rate_hz: int, language: str):
        self.track_id, self.sample_rate_hz, self.language = track_id, sample_rate_hz, language
        self.started_ms: int | None = None
        ctx = mp.get_context("spawn")
        self.commands, self.results = ctx.Queue(maxsize=16), ctx.Queue(maxsize=16)
        self.process = ctx.Process(target=_sherpa_worker, args=(model_dir, self.commands, self.results), daemon=True)
        self.process.start()
        self.last_partial = ""
        self._close_lock = threading.Lock()
        self._closed = False
        self.pending_audio = np.zeros(0, dtype=np.float32)

    def accept(self, audio: np.ndarray, sample_rate_hz: int, stamp_ms: int) -> str:
        if self.started_ms is None: self.started_ms = stamp_ms
        try:
            self.pending_audio = np.concatenate((self.pending_audio, resample_mono(audio, sample_rate_hz, 16000)))
            # 200 ms batches keep IPC and decoder scheduling overhead below
            # the realtime budget while still producing frequent partials.
            if self.pending_audio.size < 3200:
                return self._drain_partials()
            chunk, self.pending_audio = self.pending_audio.copy(), np.zeros(0, dtype=np.float32)
            self.commands.put_nowait(("accept", chunk))
        except (queue.Full, ValueError, TypeError, OSError):
            pass
        return self._drain_partials()

    def _drain_partials(self) -> str:
        # Never block PortAudio: drain whatever the worker has completed and
        # return the newest partial. The first model load can take seconds.
        try:
            while True:
                kind, value = self.results.get_nowait()
                if kind == "partial":
                    self.last_partial = str(value or "")
        except queue.Empty:
            pass
        return self.last_partial

    def finish(self, ended_ms: int) -> SpeechTranscript | None:
        value = ""
        try:
            if self.pending_audio.size:
                self.commands.put(("accept", self.pending_audio.copy()), timeout=.2)
                self.pending_audio = np.zeros(0, dtype=np.float32)
            self.commands.put(("finish", None), timeout=.2)
            deadline = time.monotonic() + 10.
            while time.monotonic() < deadline:
                kind, candidate = self.results.get(timeout=max(.1, deadline - time.monotonic()))
                if kind == "final":
                    value = str(candidate or "")
                    break
        except (queue.Empty, queue.Full, OSError, ValueError, EOFError):
            value = ""
        finally:
            self.close()
        text = value.replace(" ", "").strip()
        if not text:
            return None
        return SpeechTranscript(track_id=self.track_id, text=text, is_final=True, language=self.language,
                                confidence=0., started_ms=self.started_ms, ended_ms=ended_ms)

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self.process.join(timeout=.1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=.5)
            for channel in (self.commands, self.results):
                channel.cancel_join_thread()
                channel.close()


def _sherpa_worker(model_dir: str, commands, results) -> None:
    import sherpa_onnx
    root = Path(model_dir)
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(root / "encoder.int8.onnx"), decoder=str(root / "decoder.onnx"),
        joiner=str(root / "joiner.int8.onnx"), tokens=str(root / "tokens.txt"),
        num_threads=2, sample_rate=16000, feature_dim=80,
        decoding_method="greedy_search", provider="cpu")
    stream = recognizer.create_stream()
    while True:
        command, audio = commands.get()
        if command == "accept":
            stream.accept_waveform(16000, audio)
            while recognizer.is_ready(stream): recognizer.decode_stream(stream)
            results.put(("partial", recognizer.get_result(stream) or ""))
        else:
            stream.accept_waveform(16000, np.zeros(4800, dtype=np.float32))
            stream.input_finished()
            while recognizer.is_ready(stream): recognizer.decode_stream(stream)
            results.put(("final", recognizer.get_result(stream) or ""))
            return


class SherpaStreamingSession:
    def __init__(self, recognizer, track_id: str, sample_rate_hz: int, language: str):
        self.recognizer = recognizer
        self.stream = recognizer.create_stream()
        self.track_id = track_id
        self.sample_rate_hz = sample_rate_hz
        self.language = language
        self.started_ms: int | None = None

    def accept(self, audio: np.ndarray, sample_rate_hz: int, stamp_ms: int) -> str:
        if self.started_ms is None:
            self.started_ms = stamp_ms
        mono = resample_mono(audio, sample_rate_hz, 16000)
        self.stream.accept_waveform(16000, mono)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        return str(self.recognizer.get_result(self.stream) or "").replace(" ", "").strip()

    def finish(self, ended_ms: int) -> SpeechTranscript | None:
        # Online transducers need a short look-ahead/flush window to emit the
        # last syllables before input_finished(). Without this, a sentence can
        # end with a stale prefix (e.g. "今天") while the offline worker has
        # already produced the complete text.
        self.stream.accept_waveform(16000, np.zeros(4800, dtype=np.float32))
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        text = str(self.recognizer.get_result(self.stream) or "").replace(" ", "").strip()
        if not text:
            return None
        return SpeechTranscript(
            track_id=self.track_id,
            text=text,
            is_final=True,
            language=self.language,
            confidence=0.0,
            started_ms=self.started_ms,
            ended_ms=ended_ms,
        )


class VoskStreamingSession:
    def __init__(self, model, track_id: str, sample_rate_hz: int, language: str):
        from vosk import KaldiRecognizer

        self.track_id = track_id
        self.sample_rate_hz = sample_rate_hz
        self.language = language
        self.recognizer = KaldiRecognizer(model, 16000)
        self.recognizer.SetWords(True)
        self.started_ms: int | None = None

    def accept(self, audio: np.ndarray, sample_rate_hz: int, stamp_ms: int) -> str:
        if self.started_ms is None:
            self.started_ms = stamp_ms
        mono = resample_mono(audio, sample_rate_hz, 16000)
        pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
        self.recognizer.AcceptWaveform(pcm.tobytes())
        payload = json.loads(self.recognizer.PartialResult())
        return str(payload.get("partial") or "").replace(" ", "").strip()

    def finish(self, ended_ms: int) -> SpeechTranscript | None:
        payload = json.loads(self.recognizer.FinalResult())
        text = str(payload.get("text") or "").replace(" ", "").strip()
        if not text:
            return None
        words = payload.get("result") or []
        confidence = sum(float(item.get("conf", 0.0)) for item in words) / len(words) if words else 0.0
        return SpeechTranscript(
            track_id=self.track_id,
            text=text,
            is_final=True,
            language=self.language,
            confidence=confidence,
            started_ms=self.started_ms,
            ended_ms=ended_ms,
        )


class ExternalCommandAsrAdapter(AsrAdapter):
    """Run an external ASR command on a temporary wav file.

    The command receives `{wav}` and should print recognized text to stdout.
    This keeps local Whisper, cloud clients, or vendor SDKs outside the core
    frontend while preserving the ASR test log contract.
    """

    def __init__(
        self,
        command_template: str,
        language: str = "unknown",
        target_sample_rate_hz: int | None = None,
    ):
        self.command_template = command_template
        self.language = language
        self.target_sample_rate_hz = target_sample_rate_hz

    def transcribe(self, track_id: str, audio: np.ndarray, sample_rate_hz: int) -> SpeechTranscript | None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)
        try:
            output_audio = np.asarray(audio, dtype=np.float32)
            output_rate = sample_rate_hz
            if self.target_sample_rate_hz and self.target_sample_rate_hz != sample_rate_hz:
                output_audio = resample_mono(output_audio, sample_rate_hz, self.target_sample_rate_hz)
                output_rate = self.target_sample_rate_hz
            write_wav(wav_path, output_audio, output_rate)
            command = self.command_template.format(wav=str(wav_path))
            output = subprocess.check_output(command, shell=True, text=True).strip()
            return SpeechTranscript(
                track_id=track_id,
                text=output,
                is_final=True,
                language=self.language,
                confidence=0.0,
            )
        finally:
            wav_path.unlink(missing_ok=True)


def write_wav(path: str | Path, mono: np.ndarray, sample_rate_hz: int) -> None:
    data = np.asarray(mono, dtype=np.float32).reshape(-1)
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate_hz)
        out.writeframes(pcm.tobytes())


def resample_mono(audio: np.ndarray, source_rate_hz: int, target_rate_hz: int) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    if data.size == 0 or source_rate_hz == target_rate_hz:
        return data.copy()
    output_size = max(1, int(round(data.size * target_rate_hz / source_rate_hz)))
    source_x = np.arange(data.size, dtype=np.float64) / float(source_rate_hz)
    target_x = np.arange(output_size, dtype=np.float64) / float(target_rate_hz)
    return np.interp(target_x, source_x, data).astype(np.float32)
