from __future__ import annotations

import json
import uuid
import os
import tempfile
import threading
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

    def enrollment_embedding(self,samples,sample_rate_hz):
        audio=np.asarray(samples).reshape(-1)
        if len(audio)<sample_rate_hz*2:
            raise ValueError("请完整说至少两秒的注册话术")
        vectors=[self.embed(part,sample_rate_hz) for part in np.array_split(audio,2)]
        if any(v is None for v in vectors) or float(np.dot(vectors[0],vectors[1]))<.6:
            raise ValueError("两段声纹不一致，请一人清晰说完注册话术")
        mean=np.mean(np.asarray(vectors),axis=0)
        mean/=max(float(np.linalg.norm(mean)),1e-6)
        return mean.tolist()


class SpeakerProfileStore:
    def __init__(self, path: str | Path):
        self.lock=threading.RLock()
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
        with self.lock:
            fd,name=tempfile.mkstemp(prefix='.voices-',dir=self.path.parent)
            try:
                with os.fdopen(fd,'w') as output:json.dump(self.profiles,output,ensure_ascii=False,indent=2)
                os.replace(name,self.path)
            finally:
                if os.path.exists(name):os.unlink(name)

    def remember_stranger(self, embedding):
        speaker_id, role, similarity = self.match(embedding)
        if speaker_id != "unknown":
            return speaker_id, role, similarity
        ranked=self.rank(embedding)
        if ranked and ranked[0]["score"]>=.48:
            return "unknown","unknown",ranked[0]["score"]
        speaker_id = "stranger_" + uuid.uuid4().hex[:12]
        self.enroll(speaker_id, embedding, "stranger")
        return speaker_id, "stranger", 1.0

    def enroll(self, speaker_id: str, embedding: list[float], role: str = "known", enrollment_text: str = "") -> None:
        with self.lock:self._enroll(speaker_id,embedding,role,enrollment_text)

    def _enroll(self,speaker_id,embedding,role,enrollment_text):
        vector=self._unit(embedding)
        if vector is None:raise ValueError("invalid speaker embedding")
        embedding=vector.tolist()
        previous=self.profiles.get(str(speaker_id),{})
        aliases=list(previous.get("aliases",[]))
        pool=list(previous.get("embeddings") or ([previous["embedding"]] if previous.get("embedding") else []))
        if pool and any(self._unit(v) is None or len(v)!=len(vector) for v in pool):raise ValueError("speaker pool dimension mismatch")
        if pool and float(np.dot(vector,self._unit(pool[0])))<.55:
            raise ValueError("voice sample too far from original identity anchor")
        if pool and max(float(np.dot(vector,self._unit(v))) for v in pool if self._unit(v) is not None and len(v)==len(vector))<.6:
            raise ValueError("existing identity voice samples disagree; refusing overwrite")
        if not pool or max(float(np.dot(vector,self._unit(v))) for v in pool if self._unit(v) is not None and len(v)==len(vector))<.985:
            pool=(pool[:1]+pool[-6:]+[embedding]) if len(pool)>=8 else pool+[embedding]
        if role in {"owner", "known"}:
            query=np.asarray(embedding,dtype=np.float32)
            for old_id, old in self.profiles.items():
                vector=np.asarray(old.get("embedding",[]),dtype=np.float32)
                if old.get("role")=="stranger" and vector.shape==query.shape and float(np.dot(query,vector)/(np.linalg.norm(query)*np.linalg.norm(vector)+1e-6))>=.65:
                    old.clear()
                    old.update(canonical_id=str(speaker_id),role="alias")
                    aliases.append(old_id)
        self.profiles[str(speaker_id)] = {
            "aliases": aliases,
            "role": str(role or "known"),
            "embedding": pool[0],
            "embeddings": pool,
            "pool_version": 1,
            "reference_path":previous.get("reference_path"),
            "model": "3dspeaker-campplus-zh-cn",
            "enrollment_text": enrollment_text,
        }
        self.save()

    @staticmethod
    def _unit(embedding):
        vector=np.asarray(embedding,dtype=np.float32)
        if vector.ndim!=1 or not vector.size or not np.isfinite(vector).all():return None
        norm=float(np.linalg.norm(vector))
        return vector/norm if norm>1e-6 else None

    def rank(self,embedding):
        query=self._unit(embedding)
        if query is None:return []
        ranked=[]
        with self.lock:
            for identity,profile in self.profiles.items():
                if profile.get('role')=='alias':continue
                scores=[]
                for sample in profile.get('embeddings') or [profile.get('embedding',[])]:
                    vector=self._unit(sample)
                    if vector is not None and vector.shape==query.shape:scores.append(float(np.dot(query,vector)))
                if scores:
                    scores.sort(reverse=True)
                    score=scores[0] if len(scores)==1 else .75*scores[0]+.25*scores[1]
                    ranked.append({'speaker_id':identity,'role':profile.get('role','known'),'score':score,'pool_size':len(scores)})
        return sorted(ranked,key=lambda x:x['score'],reverse=True)

    def match(self, embedding: list[float], threshold: float = .48) -> tuple[str,str,float]:
        ranked=self.rank(embedding)
        if not ranked:return 'unknown','unknown',0.
        best=ranked[0];margin=best['score']-(ranked[1]['score'] if len(ranked)>1 else 0.)
        if best['score']<threshold or margin<.06:return 'unknown','unknown',best['score']
        return best['speaker_id'],best['role'],best['score']

    def add_verified_sample(self,identity,embedding,face_identity,face_confidence,quality):
        # Trusted, contemporaneous AV confirmation is mandatory; no self-training
        # solely from a previous nearest-neighbor voice guess.
        if identity!=face_identity or face_confidence<.8 or quality.get('overlap',1)>.15 or quality.get('echo',1)>.2 or quality.get('duration_ms',0)<2000:
            return False
        with self.lock:
            profile=self.profiles.get(identity)
            if not profile or profile.get('role') not in {'owner','known'}:return False
            ranked=self.rank(embedding)
            if not ranked or ranked[0]['speaker_id']!=identity or ranked[0]['score']<.7:return False
            if len(ranked)>1 and ranked[0]['score']-ranked[1]['score']<.12:return False
            self.enroll(identity,embedding,profile['role'],profile.get('enrollment_text',''))
            return True


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    target_size = max(1, int(round(data.size * target_rate / source_rate)))
    positions = np.linspace(0.0, max(0.0, data.size - 1), target_size)
    return np.interp(positions, np.arange(data.size), data).astype(np.float32)
