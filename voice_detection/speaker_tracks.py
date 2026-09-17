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

WINDOW_MS = 600          # embedded at once; shorter than this and the embedding is noise
HOP_MS = 300             # how often a window is taken
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

    def to_json_dict(self, stamp_ms=None):
        return {"speaker_id": self.speaker_id, "person_id": self.person_id,
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
        if samples.size < 16000 * self.window_ms // 2000:
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
                if named is None:
                    self.next_id += 1
                track = self.tracks.get(key) or SpeakerTrack(
                    speaker_id=key, person_id=named, first_heard_ms=stamp_ms, last_heard_ms=stamp_ms,
                    speech_ms=0, azimuth_deg=azimuth_deg, identity_confidence=float(similarity or 0),
                    samples=0, embedding=vector.copy())
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
            elif named and track.person_id == named:
                track.identity_confidence = max(track.identity_confidence, float(similarity or 0))
            return track

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
