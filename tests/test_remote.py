import unittest

import numpy as np

from voice_detection.remote import decode_audio_frame, decode_playback_reference, encode_audio_frame
from voice_detection.types import AudioFrame


class RemoteProtocolTest(unittest.TestCase):
    def test_audio_frame_roundtrip_keeps_shape_and_metadata(self):
        samples = np.array([[0.0, 0.25], [-0.5, 0.75]], dtype=np.float32)
        frame = AudioFrame(samples=samples, sample_rate_hz=48000, stamp_ms=123)

        restored = decode_audio_frame(encode_audio_frame(frame, "test_profile"))

        self.assertEqual(restored.sample_rate_hz, 48000)
        self.assertEqual(restored.stamp_ms, 123)
        self.assertEqual(restored.samples.shape, (2, 2))
        self.assertTrue(np.allclose(restored.samples, samples, atol=1.0 / 32768.0))

    def test_packet_can_carry_aligned_playback_reference(self):
        frame = AudioFrame(samples=np.zeros((4, 1), dtype=np.float32), sample_rate_hz=16000, stamp_ms=123)
        reference = np.asarray([0.0, 0.25, -0.5, 0.75], dtype=np.float32)

        restored = decode_playback_reference(encode_audio_frame(frame, "test_profile", reference))

        self.assertTrue(np.allclose(restored, reference, atol=1.0 / 32768.0))


if __name__ == "__main__":
    unittest.main()
