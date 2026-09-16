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


# What a startle means is "louder than people are", not "loud in dBFS". The same
# shout measures about -22 dBFS on the Mac microphone and -40.5 on the Sipeed
# array, so every absolute gate is really a statement about one device's input
# gain. The -12 dBFS floor was such a statement about a louder device: on the
# array nothing in 267 field frames ever reached it and startle never fired.
# Lowering it to -45 to match the array traded that for the opposite failure —
# ordinary speech on a normal-gain microphone clears -45 easily and every
# sentence after a silence became a startle. Both gates are now offsets from the
# level speech actually reaches on the device in use, which is a number we can
# measure per profile (MicrophoneProfile.speech_peak_dbfs).
#
# How far above speech, and for how long, comes from the startle literature.
# Human acoustic startle is elicited by stimuli around 80-105 dB(A) while
# conversational speech at working distance is 60-65, i.e. the reflex sits some
# 15-25 dB above the person talking; probability falls off sharply below that
# (Blumenthal 1988, near-threshold intensity/rise/duration; Blumenthal & Berg
# 1986, rise time and intensity). We use 14 dB rather than 15-25 because the
# comparison here is in dBFS through a microphone with automatic gain, which
# compresses exactly that range, and because the reference itself adapts below.
SPEECH_TO_STARTLE_DB = 14.
SPEECH_TO_IMMEDIATE_DB = 16.
DEFAULT_SPEECH_PEAK_DBFS = -22.
# The eliciting stimuli in those studies are 30-50 ms bursts, and the response
# grows with duration over that range (temporal summation, Blumenthal & Berg
# 1986). A single 20 ms frame is therefore not a startle — it is a click, a
# cable knock, a chair. Requiring the level to hold costs 40 ms of latency and
# removes the whole class of one-frame spikes.
STARTLE_SUSTAIN_MS = 40
# A sound near full scale is the loudest thing this device can report, so it
# stays a startle however loud the room has trained the reference to be.
STARTLE_CEILING_DBFS = -3.


@dataclass
class ReflexConfig:
    warmup_ms: int = 2500
    startle_delta_db: float = _tuned('VOICE_STARTLE_DELTA_DB', 24.)
    # The loudest level ordinary speech reaches on this microphone. Measured, not
    # assumed: read rms_dbfs on 8090 while someone talks at working distance.
    speech_peak_dbfs: float = _tuned('VOICE_SPEECH_PEAK_DBFS', DEFAULT_SPEECH_PEAK_DBFS)
    # A sanity floor only. The real discriminator is startle_delta_db, the rise
    # above the adaptive noise baseline; this stops a small noise in a very quiet
    # room from qualifying on its ratio alone. Defaults to speech peak + 10 dB.
    min_startle_dbfs: float | None = None
    # Above this the reflex stays fast; below it an impulse orients first and
    # waits for classification. Defaults to speech peak + 16 dB.
    immediate_startle_dbfs: float | None = None
    # How long the level has to hold above the floor before it counts.
    sustain_ms: int = int(_tuned('VOICE_STARTLE_SUSTAIN_MS', STARTLE_SUSTAIN_MS))
    refractory_ms: int = 8000
    orient_delta_db: float = 8.
    orient_hold_ms: int = 120
    rear_boundary_deg: float = 100.

    def __post_init__(self):
        if self.min_startle_dbfs is None:
            self.min_startle_dbfs = _tuned('VOICE_STARTLE_MIN_DBFS',
                                           self.speech_peak_dbfs + SPEECH_TO_STARTLE_DB)
        if self.immediate_startle_dbfs is None:
            self.immediate_startle_dbfs = _tuned('VOICE_STARTLE_IMMEDIATE_DBFS',
                                                 self.speech_peak_dbfs + SPEECH_TO_IMMEDIATE_DB)


def config_for_profile(profile):
    """Reflex thresholds for the microphone actually selected."""
    peak = getattr(profile, 'speech_peak_dbfs', DEFAULT_SPEECH_PEAK_DBFS)
    return ReflexConfig(speech_peak_dbfs=_tuned('VOICE_SPEECH_PEAK_DBFS', peak))


class AcousticReflex:
    def __init__(self, config=None):
        self.config = config or ReflexConfig()
        self.started_ms = None
        self.baseline = -70.
        self.previous = -120.
        self.last_startle = -10**12
        self.last_orient = -10**12
        self.orient_since = None
        self.loud_since = None
        self.loud_onset_rise = 0.
        self.speech_peak = None
        self.speech_frames = 0

    def startle_floor(self):
        """The level a sound has to reach here, now, to count as a startle.

        The configured floor is a prior about the device. What people actually
        measure through it depends on gain, distance and any automatic gain
        control, so once we have heard enough speech we take the loudest speech
        we have observed as the reference instead. It decays slowly so moving
        closer to the microphone raises the bar and moving away lowers it again.
        """
        floor = self.config.min_startle_dbfs
        if self.speech_frames >= 25 and self.speech_peak is not None:
            floor = max(floor, self.speech_peak + SPEECH_TO_STARTLE_DB)
        return min(STARTLE_CEILING_DBFS, floor)

    def update(self, dbfs, stamp_ms, direction, direction_confidence, echo=0., music=False, speech=False):
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
        if speech and clean and not music and dbfs > -60:
            self.speech_peak = dbfs if self.speech_peak is None else max(dbfs, self.speech_peak - .01)
            self.speech_frames += 1
        floor = self.startle_floor()
        qualifies = not warming and clean and not music and delta >= cfg.startle_delta_db and dbfs >= floor
        if qualifies and self.loud_since is None:
            self.loud_since = stamp_ms
            self.loud_onset_rise = rise
        elif not qualifies:
            self.loud_since = None
        # Rise is read at onset, not at the moment it fires: what the startle
        # literature requires is a fast-rising stimulus that then lasts tens of
        # milliseconds, and by the second frame the rise has already happened.
        if (self.loud_since is not None and stamp_ms - self.loud_since >= cfg.sustain_ms
                and self.loud_onset_rise >= 12 and stamp_ms - self.last_startle >= cfg.refractory_ms):
            self.last_startle = stamp_ms
            self.loud_since = None
            events.append({"kind": "startle", "stamp_ms": stamp_ms, "direction_deg": direction,
                           "direction_valid": direction is not None, "delta_db": round(delta, 2),
                           "floor_dbfs": round(floor, 1), "held_ms": cfg.sustain_ms})
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
