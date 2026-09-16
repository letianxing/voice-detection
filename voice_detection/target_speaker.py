"""Optional target-speaker extraction boundary.

The realtime path deliberately degrades to passthrough until a compatible
causal SpeakerBeam/WeSep checkpoint is configured.  This keeps VAD/DOA/ASR
stable while allowing a model backend to be added without changing ROS4HRI.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
import types
import numpy as np


@dataclass(frozen=True)
class TargetSpeechChunk:
    audio: np.ndarray
    target_probability: float
    latency_ms: float
    enabled: bool
    healthy: bool
    backend: str


class TargetSpeakerExtractor:
    def process(self, audio: np.ndarray, sample_rate_hz: int, speaker_embedding=None) -> TargetSpeechChunk:
        raise NotImplementedError

    def process_utterance(self, audio: np.ndarray, sample_rate_hz: int) -> TargetSpeechChunk:
        return self.process(audio, sample_rate_hz)


class PassthroughTargetSpeakerExtractor(TargetSpeakerExtractor):
    """Safe fallback used for normal speech and unavailable TSE models."""

    def process(self, audio: np.ndarray, sample_rate_hz: int, speaker_embedding=None) -> TargetSpeechChunk:
        del sample_rate_hz, speaker_embedding
        return TargetSpeechChunk(np.asarray(audio, dtype=np.float32).reshape(-1), 1.0, 0.0, False, True, "passthrough")

    @property
    def backend(self) -> str:
        return "baseline"


class UnavailableSpeakerBeamExtractor(PassthroughTargetSpeakerExtractor):
    @property
    def backend(self) -> str:
        return "speakerbeam-unavailable-fallback"


class CausalSpeakerBeamExtractor(PassthroughTargetSpeakerExtractor):
    """REAL-TSE causal BSRNN, executed from the non-realtime ASR worker."""

    def __init__(self, model_dir: str | Path, reference_path: str | Path):
        self.model_dir = Path(model_dir)
        self.reference_path = Path(reference_path)
        self._model = None
        self.error = ""

    @property
    def backend(self) -> str:
        return "real-tse-spk-emb-causal-100" if self.reference_path.is_file() else "speakerbeam-awaiting-reference"

    def process_utterance(self, audio: np.ndarray, sample_rate_hz: int) -> TargetSpeechChunk:
        if not self.reference_path.is_file():
            return super().process(audio, sample_rate_hz)
        started = time.perf_counter()
        try:
            import torch
            import torchaudio
            if self._model is None:
                if not hasattr(torchaudio, "set_audio_backend"):
                    torchaudio.set_audio_backend = lambda *args, **kwargs: None
                sox = types.ModuleType("torchaudio.sox_effects")
                sox.apply_effects_tensor = lambda waveform, rate, effects: (waveform, rate)
                sys.modules.setdefault("torchaudio.sox_effects", sox)
                vendor = Path(__file__).resolve().parents[1] / "vendor" / "real_tse"
                sys.path.insert(0, str(vendor))
                from wesep.cli.extractor import Extractor
                self._model = Extractor(str(self.model_dir))
            import soundfile as sf
            reference_np, reference_rate = sf.read(str(self.reference_path), dtype="float32", always_2d=True)
            reference = torch.from_numpy(reference_np.T.copy())
            mixture = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(1, -1))
            output = self._model.extract_speech_from_pcm(mixture, sample_rate_hz, reference, int(reference_rate))
            enhanced = output.detach().cpu().numpy().reshape(-1).astype(np.float32)
            return TargetSpeechChunk(enhanced, 1.0, (time.perf_counter() - started) * 1000.0, True, True, self.backend)
        except Exception as exc:
            self.error = str(exc)
            fallback = super().process(audio, sample_rate_hz)
            return TargetSpeechChunk(fallback.audio, 0.0, (time.perf_counter() - started) * 1000.0, False, False, "speakerbeam-error-fallback")


class AutoTargetSpeakerExtractor(PassthroughTargetSpeakerExtractor):
    @property
    def backend(self) -> str:
        return "auto-baseline"


def create_target_speaker_extractor(name: str = "baseline") -> TargetSpeakerExtractor:
    """Create the configured extractor; unknown/unavailable backends fall back safely."""
    if str(name).lower() == "speakerbeam":
        root = Path(__file__).resolve().parents[1]
        model_dir = root / "weights" / "real-tse" / "pretrained" / "spk_emb_causal_100"
        reference = root / "config" / "speaker_references" / "target.wav"
        if (model_dir / "avg_model.pt").is_file():
            return CausalSpeakerBeamExtractor(model_dir, reference)
        return UnavailableSpeakerBeamExtractor()
    if str(name).lower() == "auto":
        return AutoTargetSpeakerExtractor()
    return PassthroughTargetSpeakerExtractor()
