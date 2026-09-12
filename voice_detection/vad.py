from __future__ import annotations

from dataclasses import dataclass
from math import log10

import numpy as np

from .types import VoiceActivity


EPS = 1e-9


@dataclass
class EnergyVad:
    # Baseline values shared with the C++ implementation.  More permissive
    # values are not necessarily more sensitive once the adaptive noise floor
    # starts following room noise, so tuning must be opt-in and measured.
    threshold_db: float = 8.0
    min_rms_dbfs: float = -52.0
    noise_alpha: float = 0.96
    initial_noise_dbfs: float = -65.0
    calibration_frames: int = 0

    def __post_init__(self) -> None:
        self.noise_floor_dbfs = float(self.initial_noise_dbfs)
        self._calibration_seen = 0

    def process(self, mono: np.ndarray) -> VoiceActivity:
        mono = np.asarray(mono, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(mono))) + EPS)
        rms_dbfs = 20.0 * log10(max(rms, EPS))
        if self._calibration_seen < self.calibration_frames:
            if self._calibration_seen == 0:
                self.noise_floor_dbfs = rms_dbfs
            else:
                self.noise_floor_dbfs = max(self.noise_floor_dbfs, rms_dbfs)
            self._calibration_seen += 1
            return VoiceActivity(
                active=False,
                probability=0.0,
                rms_dbfs=float(rms_dbfs),
                noise_floor_dbfs=float(self.noise_floor_dbfs),
                snr_db=0.0,
            )
        snr_db = rms_dbfs - self.noise_floor_dbfs
        active = rms_dbfs >= self.min_rms_dbfs and snr_db >= self.threshold_db

        if not active:
            alpha = min(max(self.noise_alpha, 0.0), 0.999)
            self.noise_floor_dbfs = (alpha * self.noise_floor_dbfs) + ((1.0 - alpha) * rms_dbfs)

        probability = sigmoid((snr_db - self.threshold_db) / 3.0)
        if rms_dbfs < self.min_rms_dbfs:
            probability *= 0.4
        return VoiceActivity(
            active=bool(active),
            probability=float(max(0.0, min(1.0, probability))),
            rms_dbfs=float(rms_dbfs),
            noise_floor_dbfs=float(self.noise_floor_dbfs),
            snr_db=float(snr_db),
        )


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))
