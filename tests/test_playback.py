import base64
import unittest
from unittest.mock import patch, Mock
import numpy as np
from voice_detection.playback import PlaybackEngine
from voice_detection.types import AudioFrame


class PlaybackTests(unittest.TestCase):
    def test_delayed_echo_reference_is_aligned(self):
        engine = PlaybackEngine()
        rng = np.random.default_rng(7)
        audio = rng.normal(0, .05, (16000, 1)).astype(np.float32)
        engine.reference.append(AudioFrame(audio, 16000, 1000))
        # Capture at 1500ms contains playback from 1430ms: 70ms acoustic delay.
        capture = AudioFrame(audio[6880:7200] * .4, 16000, 1500)
        reference = engine.aligned_reference(capture)
        self.assertIsNotNone(reference)
        self.assertGreater(np.corrcoef(reference, capture.samples[:, 0])[0, 1], .99)

    def test_stop_id_cannot_stop_newer_playback(self):
        engine = PlaybackEngine()
        engine.playback_id = "new"
        engine.speaking = True
        engine.stream = Mock()
        engine.stop("old")
        self.assertTrue(engine.speaking)
        engine.stream.abort.assert_not_called()
        stream = engine.stream
        engine.stop("new")
        stream.abort.assert_called_once()
        self.assertFalse(engine.speaking)

    def test_pcm_validation_before_audio_device_open(self):
        engine = PlaybackEngine()
        with self.assertRaises(ValueError):
            engine.start({"id": "x", "sample_rate_hz": 16000, "pcm_s16le": base64.b64encode(b"x").decode()})
