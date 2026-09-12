import unittest

import numpy as np

from voice_detection.streaming_asr import UtteranceSegmenter


class StreamingAsrTest(unittest.TestCase):
    def test_segments_speech_after_endpoint_silence(self):
        segmenter = UtteranceSegmenter(pre_roll_ms=20, endpoint_silence_ms=40, min_speech_ms=40)
        block = np.ones(320, dtype=np.float32)

        self.assertIsNone(segmenter.process(block, 16000, 1000, False, ""))
        self.assertIsNone(segmenter.process(block, 16000, 1020, True, "voice_1"))
        self.assertIsNone(segmenter.process(block, 16000, 1040, True, "voice_1"))
        self.assertIsNone(segmenter.process(block * 0, 16000, 1060, False, ""))
        utterance = segmenter.process(block * 0, 16000, 1080, False, "")

        self.assertIsNotNone(utterance)
        self.assertEqual(utterance.track_id, "voice_1")
        self.assertEqual(utterance.started_ms, 1000)
        self.assertGreaterEqual(utterance.samples.size, 5 * block.size)

    def test_drops_too_short_speech(self):
        segmenter = UtteranceSegmenter(pre_roll_ms=0, endpoint_silence_ms=20, min_speech_ms=60)
        block = np.ones(320, dtype=np.float32)
        segmenter.process(block, 16000, 1000, True, "voice_1")

        self.assertIsNone(segmenter.process(block * 0, 16000, 1020, False, ""))


if __name__ == "__main__":
    unittest.main()
