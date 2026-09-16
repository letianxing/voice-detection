"""Frame-rate acoustic reflexes, independent of ASR and the LLM."""
import os
from collections import deque
from dataclasses import dataclass
import math
import numpy as np


def _tuned(name, fallback):
    """Field-tunable without a code change; measured defaults are for the Sipeed array."""
    try:
        return float(os.environ.get(name, fallback))
    except ValueError:
        return float(fallback)


@dataclass
class ReflexConfig:
    warmup_ms: int = 2500
    startle_delta_db: float = _tuned('VOICE_STARTLE_DELTA_DB', 24.)
    # A sanity floor only. The real discriminator is startle_delta_db, the rise
    # above the adaptive noise baseline; this just stops a small noise in a very
    # quiet room from qualifying on its ratio alone. It was -12 dBFS, which is
    # 25 dB stricter than the rise test and so decided everything by itself: on
    # the Sipeed array, whose floor measures about -61 dBFS with the loudest
    # speech frame at -40.5, nothing ever reached it and startle never fired.
    min_startle_dbfs: float = _tuned('VOICE_STARTLE_MIN_DBFS', -45.)
    refractory_ms: int = 8000
    orient_delta_db: float = 8.
    orient_hold_ms: int = 120
    rear_boundary_deg: float = 100.


class AcousticReflex:
    def __init__(self, config=None):
        self.config = config or ReflexConfig()
        self.started_ms = None
        self.baseline = -70.
        self.previous = -120.
        self.last_startle = -10**12
        self.last_orient = -10**12
        self.orient_since = None

    def update(self, dbfs, stamp_ms, direction, direction_confidence, echo=0., music=False):
        cfg = self.config
        if self.started_ms is None:
            self.started_ms, self.baseline = stamp_ms, dbfs
        delta = dbfs - self.baseline
        rise = dbfs - self.previous
        self.previous = dbfs
        warming = stamp_ms - self.started_ms < cfg.warmup_ms
        if warming or dbfs < self.baseline + 6:
            self.baseline = .97 * self.baseline + .03 * dbfs
        events = []
        clean = echo < .65
        if not warming and clean and not music and delta >= cfg.startle_delta_db and rise >= 12 and dbfs >= cfg.min_startle_dbfs and stamp_ms - self.last_startle >= cfg.refractory_ms:
            self.last_startle = stamp_ms
            events.append({"kind": "startle", "stamp_ms": stamp_ms, "direction_deg": direction,
                           "direction_valid": direction is not None, "delta_db": round(delta, 2)})
        rear = direction is not None and direction_confidence >= .45 and abs(direction) >= cfg.rear_boundary_deg
        if not warming and clean and not music and rear and delta >= cfg.orient_delta_db and dbfs >= -45:
            if self.orient_since is None:
                self.orient_since = stamp_ms
            if stamp_ms - self.orient_since >= cfg.orient_hold_ms and stamp_ms - self.last_orient >= cfg.refractory_ms:
                self.last_orient = stamp_ms
                events.append({"kind": "orient", "stamp_ms": stamp_ms, "direction_deg": direction, "direction_valid": True})
        else:
            self.orient_since = None
        return events


class LiveBeatDetector:
    """Observed onset pulses, never a free-running metronome."""
    def __init__(self):
        self.previous = None
        self.history = deque(maxlen=100)
        self.last_beat = -10**12

    def update(self, audio, stamp_ms, music_valid, bpm=None, beat_anchor_ms=None):
        spectrum = abs(np.fft.rfft(np.asarray(audio) * np.hanning(len(audio)), n=1024))
        flux = 0. if self.previous is None else float(np.maximum(0, spectrum - self.previous).sum() / (spectrum.sum() + 1e-6))
        self.previous = spectrum
        median = float(np.median(self.history)) if self.history else 0.
        mad = float(np.median(abs(np.asarray(self.history) - median))) if self.history else 0.
        threshold = max(.18, median + 3 * mad)
        self.history.append(flux)
        energy = float(np.sqrt(np.mean(np.square(audio))))
        spacing, on_grid = 200., True
        if bpm and 40 <= bpm <= 240:
            period = 60000 / bpm
            spacing = max(spacing, period * .65)
            if beat_anchor_ms is not None:
                phase_error = abs((stamp_ms - beat_anchor_ms + period / 2) % period - period / 2)
                on_grid = phase_error <= max(60., period * .22)
        pulse = music_valid and on_grid and energy > .002 and flux > threshold and stamp_ms - self.last_beat >= spacing
        if pulse:
            self.last_beat = stamp_ms
        return {"kind": "music_beat", "stamp_ms": stamp_ms, "strength": round(flux, 3)} if pulse else None
