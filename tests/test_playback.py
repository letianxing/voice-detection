import base64
import time
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


class DeadOutputStreamTests(unittest.TestCase):
    """输出流被重新初始化打死之后，不能一直说"我在播"。

    The capture supervisor re-initialises PortAudio to check whether the array
    is still on the bus, and that invalidates every stream in the process —
    including the playback output. The object survives and `active` does not,
    so warmup kept the corpse: output_warm true, render callback never running,
    first_audio_ms null, and `speaking` stuck true for ever. Brain waits for
    `speaking` to clear before it says anything else, so one dead stream ended
    the conversation. Measured on the running service: fourteen polls over 1.7
    seconds of a 0.76 second clip, first_audio_ms null the whole way.
    """

    def engine(self, samples=8000):
        from voice_detection.playback import PlaybackEngine
        import numpy as np
        engine = PlaybackEngine()
        engine.rate = 16000
        engine.samples = np.zeros(samples, np.float32)
        engine.position = 0
        engine.speaking = True
        engine.first_audio_ms = None
        engine.recent.append({"id": "p", "turn_id": "t", "text": "测试",
                              "started_ms": int(time.time() * 1000) - 900, "ended_ms": None})
        return engine

    def test_a_stall_ends_the_playback_with_an_error_instead_of_hanging(self):
        engine = self.engine()
        rebuilt = []
        engine._rebuild_output = lambda: rebuilt.append(True)
        state = engine.state()
        self.assertFalse(state["speaking"])
        self.assertIn("no audio", state["error"])
        self.assertEqual(state["stop_mode"], "stalled")
        self.assertEqual(rebuilt, [True])

    def test_a_playback_that_has_only_just_started_is_left_alone(self):
        engine = self.engine()
        engine.recent[-1]["started_ms"] = int(time.time() * 1000) - 100
        self.assertTrue(engine.state()["speaking"])
        self.assertEqual(engine.state()["error"], "")

    def test_a_playback_that_is_producing_audio_is_left_alone(self):
        engine = self.engine()
        engine.first_audio_ms = int(time.time() * 1000) - 800
        self.assertTrue(engine.state()["speaking"])
        self.assertEqual(engine.state()["error"], "")

    def test_a_dead_stream_object_is_not_mistaken_for_a_warm_one(self):
        from voice_detection.playback import PlaybackEngine
        engine = PlaybackEngine()

        class Dead:
            @property
            def active(self):
                raise RuntimeError("PortAudio not initialized")

        engine.stream = Dead()
        self.assertFalse(engine._stream_alive())

        class Live:
            active = True

        engine.stream = Live()
        self.assertTrue(engine._stream_alive())
