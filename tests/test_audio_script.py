import unittest

from voice_detection.audio_script import iter_script_payloads, load_audio_script_data, replay_audio_script


class AudioScriptTest(unittest.TestCase):
    def test_supports_direct_and_nested_payloads(self):
        plan = load_audio_script_data(
            {
                "topic": "/voice_msg_speaker",
                "steps": [
                    {"asr_text": "hello", "rate_hz": 2.0, "count": 2},
                    {"message": {"asr_text": "world"}, "period_sec": 0.25},
                ],
            }
        )

        events = list(iter_script_payloads(plan))
        self.assertEqual([item[0]["asr_text"] for item in events], ["hello", "hello", "world"])
        self.assertEqual([item[1] for item in events], [0.5, 0.5, 0.25])

    def test_replay_uses_injected_clock(self):
        plan = load_audio_script_data({"steps": [{"asr_text": "test", "count": 2, "period_sec": 0.1}]})
        emitted = []
        sleeps = []

        replay_audio_script(plan, emitted.append, sleep=sleeps.append)

        self.assertEqual(len(emitted), 2)
        self.assertEqual(sleeps, [0.1, 0.1])

    def test_rejects_empty_steps(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            load_audio_script_data({"steps": []})


if __name__ == "__main__":
    unittest.main()
