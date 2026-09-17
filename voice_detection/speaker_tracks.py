"""Who is currently talking, one entry per person, not one per microphone track.

Until now the ear produced a single dominant track per frame, so however many
people were in the room, attention saw one auditory candidate and had nothing to
choose between. Everything downstream inherited that: a cocktail-party focus
cannot switch between speakers that were never separate candidates, and a
target-speaker recogniser cannot be pointed at one of them.

What this does: every 300 ms of active speech it embeds the last 600 ms, matches
that against the enrolled profiles and against the clusters it has already seen
this session, and keeps one entry per speaker with when they were last heard and
where from. That is enough to say "A and B are both in this conversation, A spoke
last, here is a reference for each".

What it does not do: separate simultaneous speech. Two people talking at the same
instant produce one blended embedding and will land in one cluster or neither.
Separating them is what a target-speaker model is for, and that model needs
exactly what this produces — a reference per speaker. Short windows also mean the
embedding is noisier than the end-of-utterance one used for identity decisions,
so nothing here overrules the voiceprint match on a finished sentence.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

# Measured on this machine with the enrolled owner's own registration audio
# (ERes2NetV2, 2.74 s): scored against his own pool, 0.6 s windows came out
# between 0.108 and 0.708, 1.0 s windows 0.728-0.822, and 1.5 s and up
# 0.749-0.897. Below about a second the embedding says more about which
# syllable was in the window than about who said it — which is how one field
# session minted five hundred "new speakers" out of two people.
WINDOW_MS = 1500         # embedded at once
HOP_MS = 500             # how often a window is taken
MIN_EMBED_MS = 1200      # refuse to embed less speech than this
MIN_SPEECH_FOR_REFERENCE_MS = 2000   # a cluster has to survive before it earns a file
ACTIVE_MS = 3000         # a speaker stays "in the conversation" this long after speaking
FORGET_MS = 120000       # and is dropped from the session entirely after this
NEW_CLUSTER_BELOW = .55  # below this similarity to every known cluster, it is somebody new
JOIN_ABOVE = .62         # at or above this, it is that cluster
MAX_CLUSTERS = 8


@dataclass
class SpeakerTrack:
    speaker_id: str
    person_id: str | None
    first_heard_ms: int
    last_heard_ms: int
    speech_ms: int
    azimuth_deg: float | None
    identity_confidence: float
    samples: int
    embedding: np.ndarray = field(repr=False, default=None)
    reference_path: str | None = None
    reference_quality: float = 0.0
    display_name: str | None = None

    def digest(self, buckets=24):
        """A small, stable summary of the voiceprint for the console.

        192 numbers do not fit on a screen and mean nothing to a reader; the
        point of showing them is to see at a glance that two rows are different
        voices. Mean-pooled into two dozen buckets they still look different
        when they are different, and identical when it is one person.
        """
        vector = _unit(self.embedding)
        if vector is None:
            return []
        groups = np.array_split(vector, min(buckets, vector.size))
        return [round(float(part.mean()), 3) for part in groups]

    def to_json_dict(self, stamp_ms=None):
        return {"speaker_id": self.speaker_id, "person_id": self.person_id,
                "display_name": self.display_name or self.person_id or self.speaker_id,
                "embedding_digest": self.digest(),
                "reference_available": bool(self.reference_path),
                "first_heard_ms": self.first_heard_ms, "last_heard_ms": self.last_heard_ms,
                "speech_ms": self.speech_ms, "azimuth_deg": self.azimuth_deg,
                "identity_confidence": round(float(self.identity_confidence), 3),
                "windows": self.samples,
                "age_ms": None if stamp_ms is None else max(0, int(stamp_ms) - self.last_heard_ms)}


class MixedAudioRing:
    """The last few seconds of microphone audio, kept for looking backwards.

    Focused hearing has to be able to start mid-sentence: attention decides to
    listen to somebody a beat after they started speaking, and the beginning of
    the sentence is already gone unless it was kept. Nothing is written to disk.
    """

    def __init__(self, sample_rate_hz=16000, seconds=30.0):
        self.rate = int(sample_rate_hz)
        self.capacity = int(self.rate * float(seconds))
        self.lock = threading.Lock()
        self.buffer = np.zeros(0, dtype=np.float32)
        self.end_ms = 0

    def append(self, samples, stamp_ms):
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not block.size:
            return
        with self.lock:
            self.buffer = np.concatenate((self.buffer, block))[-self.capacity:]
            self.end_ms = int(stamp_ms) + int(block.size * 1000 / self.rate)

    def window(self, duration_ms, until_ms=None):
        """The most recent duration_ms of audio ending at until_ms (default: now)."""
        with self.lock:
            if self.buffer.size == 0:
                return np.zeros(0, dtype=np.float32), 0
            end = self.end_ms if until_ms is None else min(int(until_ms), self.end_ms)
            trailing_ms = max(0, self.end_ms - end)
            stop = self.buffer.size - int(trailing_ms * self.rate / 1000)
            start = max(0, stop - int(duration_ms * self.rate / 1000))
            if stop <= start:
                return np.zeros(0, dtype=np.float32), end
            return self.buffer[start:stop].copy(), end

    def span_ms(self):
        with self.lock:
            return int(self.buffer.size * 1000 / self.rate)


def _unit(vector):
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 1e-6 else None


class SpeakerTracker:
    """Online clustering of short speech windows into one entry per speaker."""

    def __init__(self, embedder, profiles=None, window_ms=WINDOW_MS, hop_ms=HOP_MS,
                 join_above=JOIN_ABOVE, new_below=NEW_CLUSTER_BELOW):
        self.embedder = embedder
        self.profiles = profiles
        self.window_ms = int(window_ms)
        self.hop_ms = int(hop_ms)
        self.join_above = float(join_above)
        self.new_below = float(new_below)
        self.lock = threading.Lock()
        self.tracks: dict[str, SpeakerTrack] = {}
        self.next_id = 1
        self.last_window_ms = 0
        self.last_reason = "no_window_yet"
        self.embed_ms = 0.0

    def due(self, stamp_ms):
        return int(stamp_ms) - self.last_window_ms >= self.hop_ms

    def observe(self, audio, stamp_ms, azimuth_deg=None, echo_probability=0.0):
        """Embed one window and fold it into the speaker set. Returns the track."""
        self.last_window_ms = int(stamp_ms)
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size < 16 * max(MIN_EMBED_MS, self.window_ms // 2):
            self.last_reason = "window_too_short"
            return None
        if echo_probability >= .3:
            self.last_reason = "our_own_playback"
            return None
        if float(np.sqrt(np.mean(np.square(samples)))) < 1e-4:
            self.last_reason = "silent_window"
            return None
        started = time.perf_counter()
        try:
            embedding = self.embedder.embed(samples, 16000)
        except Exception as exc:
            self.last_reason = f"embedder_failed:{exc}"
            return None
        self.embed_ms = (time.perf_counter() - started) * 1000
        vector = _unit(embedding) if embedding is not None else None
        if vector is None:
            self.last_reason = "no_embedding"
            return None
        return self._assign(vector, int(stamp_ms), azimuth_deg)

    def _assign(self, vector, stamp_ms, azimuth_deg):
        with self.lock:
            self._forget(stamp_ms)
            best_id, best_score = None, 0.0
            for key, track in self.tracks.items():
                saved = _unit(track.embedding)
                if saved is None or saved.shape != vector.shape:
                    continue
                score = float(vector @ saved)
                if score > best_score:
                    best_id, best_score = key, score

            named, role, similarity = (None, None, 0.0)
            if self.profiles is not None:
                try:
                    named, role, similarity = self.profiles.match(vector.tolist())
                except Exception:
                    named, role, similarity = (None, None, 0.0)
            if named in {"unknown", ""}:
                named = None

            if best_id is not None and best_score >= self.join_above:
                track = self.tracks[best_id]
                self.last_reason = "joined_existing"
            elif best_id is not None and best_score >= self.new_below:
                # Between the two bars: probably the same person, said so in the
                # payload rather than silently minting a second copy of them.
                track = self.tracks[best_id]
                self.last_reason = "joined_uncertain"
            else:
                key = named or f"spk_{self.next_id:02d}"
                ordinal = self.next_id
                if named is None:
                    self.next_id += 1
                track = self.tracks.get(key) or SpeakerTrack(
                    speaker_id=key, person_id=named, first_heard_ms=stamp_ms, last_heard_ms=stamp_ms,
                    speech_ms=0, azimuth_deg=azimuth_deg, identity_confidence=float(similarity or 0),
                    samples=0, embedding=vector.copy(),
                    display_name=self._name_for(named) or f"声音{ordinal}")
                self.tracks[key] = track
                self.last_reason = "new_speaker"
                if len(self.tracks) > MAX_CLUSTERS:
                    oldest = min(self.tracks, key=lambda name: self.tracks[name].last_heard_ms)
                    self.tracks.pop(oldest, None)

            track.last_heard_ms = stamp_ms
            track.speech_ms += self.hop_ms
            track.samples += 1
            track.embedding = _unit(track.embedding * .8 + vector * .2) if track.embedding is not None else vector
            if azimuth_deg is not None:
                track.azimuth_deg = azimuth_deg if track.azimuth_deg is None else \
                    .7 * track.azimuth_deg + .3 * azimuth_deg
            if named and not track.person_id:
                # The enrolled match is the one that names a cluster; a cluster
                # never renames itself from its own drifting average.
                track.person_id = named
                track.identity_confidence = float(similarity or 0)
                track.display_name = self._name_for(named) or track.display_name
            elif named and track.person_id == named:
                track.identity_confidence = max(track.identity_confidence, float(similarity or 0))
                track.display_name = self._name_for(named) or track.display_name
            return track

    def _name_for(self, identity):
        """The registered person's own name, or the ordinal they were given."""
        if not identity:
            return None
        resolve = getattr(self.profiles, "display_name", None)
        try:
            return str(resolve(identity)) if callable(resolve) else str(identity)
        except Exception:
            return str(identity)

    def _forget(self, stamp_ms):
        for key in [key for key, track in self.tracks.items() if stamp_ms - track.last_heard_ms > FORGET_MS]:
            self.tracks.pop(key, None)

    def active(self, stamp_ms, within_ms=ACTIVE_MS):
        with self.lock:
            return [track for track in sorted(self.tracks.values(), key=lambda t: -t.last_heard_ms)
                    if 0 <= int(stamp_ms) - track.last_heard_ms <= within_ms]

    def state(self, stamp_ms):
        with self.lock:
            everyone = sorted(self.tracks.values(), key=lambda t: -t.last_heard_ms)
        return {"speakers": [track.to_json_dict(stamp_ms) for track in everyone],
                "active": [track.speaker_id for track in self.active(stamp_ms)],
                "reason": self.last_reason, "embed_ms": round(self.embed_ms, 1),
                "window_ms": self.window_ms, "hop_ms": self.hop_ms}


class SpeakerReferenceResolver:
    """Which recording to hand a target-speaker recogniser for this person.

    Two places to look. An enrolled identity already has a reference saved from
    registration, kept in the speaker profile store. Anybody else — a visitor
    the room has only just met — has nothing, and asking somebody to enrol
    before the robot can follow them in a crowd is not how any of this should
    work, so a clean window of their own speech is captured as they talk.

    No reference means no target-speaker ASR. Handing the model somebody else's
    voice, or a two-second scrap of overlapping noise, produces a confident
    transcript of the wrong person.
    """

    MIN_SECONDS = 2.0
    MAX_SECONDS = 6.0

    def __init__(self, profiles, folder, ring=None):
        from pathlib import Path
        self.profiles = profiles
        self.folder = Path(folder)
        self.ring = ring
        self.session = {}
        self.last_reason = "no_request_yet"

    def enrolled_reference(self, person_id):
        from pathlib import Path
        profile = (self.profiles.profiles.get(str(person_id)) if self.profiles else None) or {}
        path = profile.get("reference_path")
        return str(path) if path and Path(path).is_file() else None

    def resolve(self, speaker_id, person_id=None):
        """The best reference for this speaker, or None with a reason recorded."""
        for candidate in (person_id, speaker_id):
            if not candidate:
                continue
            path = self.enrolled_reference(candidate)
            if path:
                self.last_reason = "enrolled_reference"
                return path
        path = self.session.get(str(speaker_id))
        if path:
            self.last_reason = "session_reference"
            return path
        self.last_reason = "no_reference_yet"
        return None

    def capture(self, track, quality, write_wav=None):
        """Keep a clean window of this speaker as their session reference.

        Only from audio good enough to be worth conditioning a model on: no
        overlap, no playback of our own, and long enough to carry a voice.
        """
        import wave
        from pathlib import Path
        if self.ring is None or track is None:
            return None
        if quality.get("overlap", 1) > .2 or quality.get("echo", 1) > .2:
            self.last_reason = "window_not_clean"
            return None
        if int(track.speech_ms or 0) < MIN_SPEECH_FOR_REFERENCE_MS:
            # One window is not a speaker. Writing a file for every cluster the
            # moment it appeared left five hundred six-second wavs on disk from
            # a session with two people in the room.
            self.last_reason = "cluster_too_new"
            return None
        existing = self.session.get(track.speaker_id)
        score = float(1.0 - quality.get("overlap", 0)) * float(track.identity_confidence or .5)
        if existing and score <= track.reference_quality:
            return existing
        audio, _ = self.ring.window(int(self.MAX_SECONDS * 1000))
        if audio.size < int(self.MIN_SECONDS * 16000):
            self.last_reason = "not_enough_audio"
            return None
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"session_{track.speaker_id}.wav"
        if write_wav is None:
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000)
                handle.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        else:
            write_wav(path, audio, 16000)
        self.session[track.speaker_id] = str(path)
        track.reference_path = str(path)
        track.reference_quality = score
        self.last_reason = "captured_session_reference"
        return str(path)


class AuditoryAttentionTarget:
    """Who attention has decided to listen to, and until when.

    Held here rather than inferred, because the decision belongs upstream: the
    ear does not choose whose voice matters. An expired target is not a target —
    focused hearing falls back to whoever is loudest rather than carrying on with
    somebody who left.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.target = None

    def set(self, payload, stamp_ms):
        with self.lock:
            self.target = {
                "speaker_id": str(payload.get("speaker_id") or ""),
                "person_id": str(payload.get("person_id") or "") or None,
                "reference_id": payload.get("reference_id"),
                "direction_deg": payload.get("direction_deg"),
                "strength": float(payload.get("strength") or 1.0),
                "ttl_ms": int(payload.get("ttl_ms") or 4000),
                "pre_roll_ms": int(payload.get("pre_roll_ms") or 2000),
                "reason": str(payload.get("reason") or ""),
                "issued_ms": int(stamp_ms),
            }
            return dict(self.target)

    def current(self, stamp_ms):
        with self.lock:
            target = self.target
            if not target:
                return None
            if int(stamp_ms) - target["issued_ms"] > target["ttl_ms"]:
                return None
            return dict(target, age_ms=int(stamp_ms) - target["issued_ms"])
