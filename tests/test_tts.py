import unittest

from voice_detection.tts import speak_text


class TtsTest(unittest.TestCase):
    def test_dry_run_returns_playback_state(self):
        state = speak_text("测试", dry_run=True)

        self.assertFalse(state.speaking)
        self.assertEqual(state.text, "测试")
        self.assertIn("dry_run", state.backend)


if __name__ == "__main__":
    unittest.main()
