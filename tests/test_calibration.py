import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from voice_detection.calibration import analyze_wav
from voice_detection.profiles import get_profile
from voice_detection.simulation import synthetic_array_frame


class WavCalibrationTest(unittest.TestCase):
    def test_analyze_sipeed_like_multichannel_wav(self):
        profile = get_profile("sipeed_6_plus_1_usb_array")
        frame = synthetic_array_frame(profile, azimuth_deg=32.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "array.wav"
            _write_multichannel_wav(path, frame.samples, frame.sample_rate_hz)

            report = analyze_wav(path, profile_id=profile.id)
            payload = json.loads(report.to_json())

        self.assertEqual(payload["channel_count"], profile.input_channels)
        self.assertEqual(payload["sample_rate_hz"], profile.sample_rate_hz)
        self.assertEqual(len(payload["channel_metrics"]), profile.input_channels)
        self.assertEqual(len(payload["correlation_matrix"]), profile.input_channels)
        self.assertEqual(payload["warnings"], [])
        self.assertIsNotNone(payload["doa_estimate"])
        self.assertGreater(payload["doa_estimate"]["confidence"], 0.1)

    def test_analyze_reports_profile_mismatch(self):
        samples = np.zeros((1600, 1), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mono.wav"
            _write_multichannel_wav(path, samples, 16000)
            report = analyze_wav(path, profile_id="sipeed_6_plus_1_usb_array")

        self.assertIn("channel_count_mismatch", " ".join(report.warnings))
        self.assertIn("sample_rate_mismatch", " ".join(report.warnings))


def _write_multichannel_wav(path: Path, samples: np.ndarray, sample_rate_hz: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(samples.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm.tobytes())


if __name__ == "__main__":
    unittest.main()
