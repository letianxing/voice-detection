import json
import unittest

import numpy as np

from voice_detection.audio_frame_codec import (
    decode_audio_frame_payload,
    encode_audio_frame_payload,
    float32_to_pcm16,
    pcm16_to_float32,
)
from voice_detection.pacific_rim_compat import acoustic_track_from_audio_msg, audio_msg_from_acoustic_track


class PacificRimCompatTest(unittest.TestCase):
    def test_audio_msg_maps_to_track_and_transcript(self):
        track = acoustic_track_from_audio_msg(
            {
                "user_id": "owner",
                "internal_id": "script-001",
                "role": "tester",
                "asr_text": "hello from script",
                "emotion": "neutral",
                "angle": 12.5,
                "speaker_vector": [0.1, 0.2],
            },
            stamp_ms=1000,
        )

        self.assertEqual(track.track_id, "script-001")
        self.assertEqual(track.speaker_label, "owner")
        self.assertEqual(track.azimuth_deg, 12.5)
        self.assertTrue(track.voice_activity)
        self.assertEqual(track.transcript.text, "hello from script")

    def test_track_maps_back_to_audio_msg_shape(self):
        track = acoustic_track_from_audio_msg({"user_id": "owner", "asr_text": "你好", "angle": -20.0}, stamp_ms=1000)
        msg = audio_msg_from_acoustic_track(track)

        self.assertEqual(msg["user_id"], "owner")
        self.assertEqual(msg["asr_text"], "你好")
        self.assertEqual(msg["angle"], -20.0)
        self.assertEqual(msg["speaker_vector"], [])

    def test_tts_audio_frame_payload_round_trip(self):
        samples = np.asarray([[0.0, 0.1], [-0.2, 0.3]], dtype=np.float32)

        payload = encode_audio_frame_payload(samples, 16000, "tts-frame", stamp_sec=1, stamp_nanosec=500_000_000)
        decoded_json = json.loads(payload.decode("utf-8"))
        frame = decode_audio_frame_payload(decoded_json)

        self.assertEqual(frame.sample_rate_hz, 16000)
        self.assertEqual(frame.stamp_ms, 1500)
        self.assertEqual(frame.samples.shape, (2, 2))
        np.testing.assert_allclose(frame.samples, samples, atol=1e-6)

    def test_pcm16_conversion_is_bounded(self):
        pcm = float32_to_pcm16(np.asarray([-2.0, 0.0, 2.0], dtype=np.float32))
        restored = pcm16_to_float32(pcm)

        self.assertEqual(pcm.dtype, np.int16)
        self.assertGreaterEqual(restored.min(), -1.0)
        self.assertLessEqual(restored.max(), 1.0)


if __name__ == "__main__":
    unittest.main()
