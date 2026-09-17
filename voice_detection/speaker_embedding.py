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


# Silent enrollment of strangers. A single utterance is not evidence of a person:
# the stranger profile starts as provisional, is grown by later utterances that
# match it, and is only published once it has accumulated enough. Values marked
# "pending calibration" were measured on CAM++ (CN-Celeb) and must be rescanned
# for ERes2NetV2 before being trusted; see docs in the pacific-rim voiceprint repo.
STRANGER_MIN_DURATION_MS = 4000          # single utterance shorter than this cannot open a profile
STRANGER_GRAY_FLOOR = 0.50               # best score in [floor, match threshold): too close to an existing
                                         # voice to open a new profile, not close enough to accept. Pending calibration.
STRANGER_PROMOTE_MIN_SAMPLES = 3
STRANGER_PROMOTE_MIN_DURATION_MS = 15000
STRANGER_PROVISIONAL_TTL_MS = 72 * 3600 * 1000
STRANGER_DEGENERATE_MEDIAN = 0.93        # pool pairwise-median above this is a collapsed embedding, not a voice. Pending calibration.
LIFECYCLE_KEYS = ("lifecycle", "created_ms", "samples", "total_duration_ms", "evidence_similarity", "evidence_bodies")


def _body_conflict(bodies):
    """Two utterances credited to different visible people while both were in view.

    Each entry is {"internal_id": who the utterance was bound to, "visible_ids": everyone
    in the camera at that moment}. One voice cannot come from two bodies that the
    camera saw at the same time, so the profile is not one person. Sequential ids are
    not a conflict: an unregistered face gets a new track id whenever it leaves and
    returns, so A-then-B is usually the same person renumbered.
    """
    known = [b for b in bodies if b.get("internal_id")]
    for i in range(len(known)):
        for j in range(i + 1, len(known)):
            a, b = known[i], known[j]
            if a["internal_id"] == b["internal_id"]:
                continue
            if a["internal_id"] in (b.get("visible_ids") or []) or b["internal_id"] in (a.get("visible_ids") or []):
                return a["internal_id"], b["internal_id"]
    return None


class SpeakerProfileStore:
    def __init__(self, path: str | Path):
        self.lock=threading.RLock()
        self.path = Path(path).expanduser()
        self.profiles: dict[str, dict[str, Any]] = {}
        self.load()

    def lifecycle(self, speaker_id) -> str:
        """Profiles written before lifecycle existed are established: they were enrolled deliberately."""
        profile = self.profiles.get(str(speaker_id), {})
        return str(profile.get("lifecycle") or "established")

    def publishable(self, speaker_id) -> bool:
        return self.lifecycle(speaker_id) == "established"

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

    def remember_stranger(self, embedding, duration_ms=None, now_ms=None, internal_id="", visible_ids=None):
        """Attribute an unregistered utterance without publishing a guess.

        Returns (speaker_id, role, similarity). The id is "unknown" unless the
        utterance matched a published profile; a provisional stranger absorbs the
        utterance as evidence and stays "unknown" until promoted.
        """
        now_ms = int(now_ms if now_ms is not None else __import__("time").time() * 1000)
        with self.lock:
            speaker_id, role, similarity = self._match_any(embedding)
            if speaker_id != "unknown":
                if self.profiles.get(speaker_id, {}).get("role") == "stranger":
                    return self._grow_stranger(speaker_id, embedding, duration_ms, now_ms, similarity,
                                               internal_id, visible_ids)
                return speaker_id, role, similarity
            ranked = self.rank(embedding)
            best = ranked[0]["score"] if ranked else 0.0
            # Gray zone: resembles an existing voice but not enough to accept. Opening a
            # profile here would split one person across two ids on their next sentence.
            if best >= STRANGER_GRAY_FLOOR:
                return "unknown", "unknown", best
            if duration_ms is not None and int(duration_ms) < STRANGER_MIN_DURATION_MS:
                return "unknown", "unknown", best
            self._expire_provisional(now_ms)
            speaker_id = "stranger_" + uuid.uuid4().hex[:12]
            self._enroll(speaker_id, embedding, "stranger", "")
            self.profiles[speaker_id].update(lifecycle="provisional", created_ms=now_ms, samples=1,
                                             total_duration_ms=int(duration_ms or 0),
                                             evidence_bodies=[self._body_entry(internal_id, visible_ids)])
            self.save()
            return "unknown", "unknown", best

    @staticmethod
    def _body_entry(internal_id, visible_ids):
        return {"internal_id": str(internal_id or ""),
                "visible_ids": [str(v) for v in (visible_ids or []) if str(v)]}

    def _grow_stranger(self, speaker_id, embedding, duration_ms, now_ms, similarity, internal_id="", visible_ids=None):
        profile = self.profiles[speaker_id]
        state = self.lifecycle(speaker_id)
        if state == "established":
            # Still watched: a published profile hit from a second visible body is withdrawn.
            bodies = (list(profile.get("evidence_bodies") or []) + [self._body_entry(internal_id, visible_ids)])[-8:]
            profile["evidence_bodies"] = bodies
            if _body_conflict(bodies):
                profile["lifecycle"] = "contested"
                self.save()
                return "unknown", "unknown", similarity
            self.save()
            return speaker_id, "stranger", similarity
        if state != "provisional":
            return "unknown", "unknown", similarity
        query = self._unit(embedding)
        pool = [self._unit(v) for v in (profile.get("embeddings") or [])]
        closest = max((float(np.dot(query, v)) for v in pool if v is not None and query is not None
                       and v.shape == query.shape), default=0.0)
        try:
            self._enroll(speaker_id, embedding, "stranger", "")
        except ValueError:
            # Too far from the anchor or the pool disagrees: not evidence for this profile.
            return "unknown", "unknown", similarity
        profile = self.profiles[speaker_id]
        profile["samples"] = int(profile.get("samples") or 0) + 1
        profile["total_duration_ms"] = int(profile.get("total_duration_ms") or 0) + int(duration_ms or 0)
        # How close each new utterance sat to what was already there. Real voices vary
        # between sentences; a run of near-identical vectors is a collapsed embedding
        # (seen once in field data: three people merged into one "very consistent" profile).
        evidence = list(profile.get("evidence_similarity") or []) + [round(closest, 4)]
        profile["evidence_similarity"] = evidence[-8:]
        bodies = (list(profile.get("evidence_bodies") or []) + [self._body_entry(internal_id, visible_ids)])[-8:]
        profile["evidence_bodies"] = bodies
        if _body_conflict(bodies):
            profile["lifecycle"] = "contested"
        elif len(evidence) >= 2 and float(np.median(evidence)) > STRANGER_DEGENERATE_MEDIAN:
            profile["lifecycle"] = "frozen"
        elif (profile["samples"] >= STRANGER_PROMOTE_MIN_SAMPLES
              and profile["total_duration_ms"] >= STRANGER_PROMOTE_MIN_DURATION_MS):
            profile["lifecycle"] = "established"
        self.save()
        if profile["lifecycle"] == "established":
            return speaker_id, "stranger", similarity
        return "unknown", "unknown", similarity

    def _expire_provisional(self, now_ms):
        stale = [sid for sid, p in self.profiles.items()
                 if p.get("role") == "stranger" and p.get("lifecycle") == "provisional"
                 and p.get("created_ms") is not None
                 and now_ms - int(p["created_ms"]) > STRANGER_PROVISIONAL_TTL_MS]
        for sid in stale:
            self.profiles.pop(sid, None)

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
            **{key: previous[key] for key in LIFECYCLE_KEYS if key in previous},
            "aliases": aliases,
            "role": str(role or "known"),
            "embedding": pool[0],
            "embeddings": pool,
            "pool_version": 1,
            "reference_path":previous.get("reference_path"),
            "model": "3dspeaker-eres2netv2-zh-cn",
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

    def _match_any(self, embedding: list[float], threshold: float = .48) -> tuple[str,str,float]:
        ranked=self.rank(embedding)
        if not ranked:return 'unknown','unknown',0.
        best=ranked[0];margin=best['score']-(ranked[1]['score'] if len(ranked)>1 else 0.)
        if best['score']<threshold or margin<.06:return 'unknown','unknown',best['score']
        return best['speaker_id'],best['role'],best['score']

    def match(self, embedding: list[float], threshold: float = .48) -> tuple[str,str,float]:
        """Published identity: a provisional or frozen stranger matches internally but is reported unknown."""
        speaker_id, role, score = self._match_any(embedding, threshold)
        if speaker_id != 'unknown' and not self.publishable(speaker_id):
            return 'unknown', 'unknown', score
        return speaker_id, role, score

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
