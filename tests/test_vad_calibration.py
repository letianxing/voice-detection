import unittest

import numpy as np

from voice_detection.vad import EnergyVad


class VadCalibrationTest(unittest.TestCase):
    def test_warmup_learns_ambient_level_before_opening_gate(self):
        vad = EnergyVad(calibration_frames=3)
        ambient = np.ones(960, dtype=np.float32) * 0.004

        for _ in range(3):
            self.assertFalse(vad.process(ambient).active)
        self.assertFalse(vad.process(ambient).active)
        self.assertTrue(vad.process(ambient * 6.0).active)

    def test_startup_zero_frames_do_not_poison_noise_floor(self):
        vad = EnergyVad(calibration_frames=3)
        silence = np.zeros(960, dtype=np.float32)
        ambient = np.ones(960, dtype=np.float32) * 0.004

        vad.process(silence)
        vad.process(ambient)
        vad.process(ambient)

        self.assertFalse(vad.process(ambient).active)


if __name__ == "__main__":
    unittest.main()
