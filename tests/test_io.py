import tempfile
import unittest
from pathlib import Path

import numpy as np

from voice_detection.io import read_wav, write_wav
from voice_detection.types import AudioFrame


class AudioIoTest(unittest.TestCase):
    def test_multichannel_wav_roundtrip(self):
        samples = np.linspace(-0.5, 0.5, 800, dtype=np.float32).reshape(100, 8)
        frame = AudioFrame(samples=samples, sample_rate_hz=48000, stamp_ms=1000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.wav"
            write_wav(path, frame)
            restored = read_wav(path)

        self.assertEqual(restored.samples.shape, (100, 8))
        self.assertEqual(restored.sample_rate_hz, 48000)
        np.testing.assert_allclose(restored.samples, samples, atol=2.0 / 32767.0)


if __name__ == "__main__":
    unittest.main()
