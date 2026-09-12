from __future__ import annotations

from math import log10

import numpy as np


class AdaptiveGainControl:
    """Conservative digital AGC for weak far-field speech.

    It only raises quiet frames, never boosts clipped input, and changes gain
    slowly enough that the noise floor does not jump on every 20 ms block.
    """

    def __init__(
        self,
        target_dbfs: float = -32.0,
        max_gain_db: float = 6.0,
        attack: float = 0.08,
        release: float = 0.02,
    ):
        self.target_dbfs = float(target_dbfs)
        self.max_gain_db = float(max_gain_db)
        self.attack = float(np.clip(attack, 0.01, 1.0))
        self.release = float(np.clip(release, 0.005, 1.0))
        self.gain_db = 0.0

    def process(self, mono: np.ndarray) -> np.ndarray:
        data = np.asarray(mono, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return data
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(np.square(data))) + 1e-9)
        rms_dbfs = 20.0 * log10(max(rms, 1e-9))
        # Do not chase silence: this is the main protection against pumping
        # the Sipeed room noise into the VAD/ASR path.
        if rms_dbfs < -70.0:
            desired_db = 0.0
            coefficient = self.release
            self.gain_db += coefficient * (desired_db - self.gain_db)
            return data.copy()
        if peak >= 0.98:
            desired_db = min(0.0, self.gain_db)
        else:
            desired_db = float(np.clip(self.target_dbfs - rms_dbfs, 0.0, self.max_gain_db))
        coefficient = self.attack if desired_db > self.gain_db else self.release
        self.gain_db += coefficient * (desired_db - self.gain_db)
        gain = 10.0 ** (self.gain_db / 20.0)
        return np.clip(data * gain, -1.0, 1.0).astype(np.float32)
