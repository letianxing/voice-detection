from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class SpeakerEmbedder:
    """Optional sherpa-onnx speaker encoder with cosine matching."""

    def __init__(self, model_path: str | Path, num_threads: int = 2):
        self.model_path = Path(model_path).expanduser()
        self.extractor = None
        self.error = ""
        try:
            import sherpa_onnx

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.model_path),
                num_threads=int(num_threads),
                provider="cpu",
            )
            self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        except Exception as exc:
            self.error = str(exc)

    @property
    def ready(self) -> bool:
        return self.extractor is not None

    @property
    def dim(self) -> int:
        return int(self.extractor.dim) if self.extractor is not None else 0

    def embed(self, samples: np.ndarray, sample_rate_hz: int) -> list[float] | None:
        if self.extractor is None:
            return None
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size < max(1600, int(sample_rate_hz * 0.5)):
            return None
        if sample_rate_hz != 16000:
            audio = resample(audio, sample_rate_hz, 16000)
        stream = self.extractor.create_stream()
        stream.accept_waveform(16000, audio)
        if not self.extractor.is_ready(stream):
            return None
        vector = np.asarray(self.extractor.compute(stream), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return (vector / norm).tolist() if norm > 1e-6 else None


class SpeakerProfileStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.profiles: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.profiles = payload if isinstance(payload, dict) else {}
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            self.profiles = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.profiles, ensure_ascii=False, indent=2), encoding="utf-8")

    def enroll(self, speaker_id: str, embedding: list[float], role: str = "known") -> None:
        self.profiles[str(speaker_id)] = {
            "role": str(role or "known"),
            "embedding": list(embedding),
            "model": "3dspeaker-campplus-zh-cn",
        }
        self.save()

    def match(self, embedding: list[float], threshold: float = 0.48) -> tuple[str, str, float]:
        query = np.asarray(embedding, dtype=np.float32)
        best_id, best_role, best_score = "unknown", "unknown", 0.0
        for speaker_id, profile in self.profiles.items():
            candidate = np.asarray(profile.get("embedding", []), dtype=np.float32)
            if candidate.size != query.size:
                continue
            score = float(np.dot(query, candidate) / (np.linalg.norm(candidate) + 1e-6))
            if score > best_score:
                best_id, best_role, best_score = speaker_id, str(profile.get("role") or "known"), score
        if best_score < float(threshold):
            return "unknown", "unknown", best_score
        return best_id, best_role, best_score


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    target_size = max(1, int(round(data.size * target_rate / source_rate)))
    positions = np.linspace(0.0, max(0.0, data.size - 1), target_size)
    return np.interp(positions, np.arange(data.size), data).astype(np.float32)
