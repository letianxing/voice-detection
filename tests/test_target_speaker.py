import unittest
import numpy as np

from voice_detection.target_speaker import PassthroughTargetSpeakerExtractor


class TargetSpeakerExtractorTest(unittest.TestCase):
    def test_passthrough_is_safe_fallback(self):
        audio = np.arange(8, dtype=np.float32)
        result = PassthroughTargetSpeakerExtractor().process(audio, 16000)
        np.testing.assert_array_equal(result.audio, audio)
        self.assertTrue(result.healthy)
        self.assertFalse(result.enabled)
        self.assertEqual(result.backend, "passthrough")


if __name__ == "__main__":
    unittest.main()
