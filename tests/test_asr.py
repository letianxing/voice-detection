import json
import unittest

import numpy as np

from voice_detection.asr import AsrScenarioRecord, resample_mono
from voice_detection.types import SpeechTranscript


class AsrBoundaryTest(unittest.TestCase):
    def test_asr_record_keeps_test_boundary_fields(self):
        record = AsrScenarioRecord(
            scenario_id="negation",
            utterance_expected="别关灯",
            environment={"distance_m": 1.5, "relative_direction_deg": 0},
            speaking_style={"volume_dba": 62, "language_mix": "zh-CN"},
            asr_result=SpeechTranscript(
                track_id="voice_1",
                text="别关灯",
                is_final=True,
                language="zh-CN",
                clarity=0.8,
                started_ms=100,
                ended_ms=700,
                emitted_ms=1100,
            ),
            diff_notes={"missing_keywords": []},
        )

        payload = json.loads(record.to_json())

        self.assertEqual(payload["asr_result"]["text"], "别关灯")
        self.assertEqual(payload["asr_result"]["track_id"], "voice_1")
        self.assertIn("environment", payload)
        self.assertIn("speaking_style", payload)
        self.assertNotIn("intent", payload)
        self.assertNotIn("permission", payload)

    def test_resamples_capture_audio_for_local_asr(self):
        source = np.linspace(-0.5, 0.5, 4800, dtype=np.float32)

        output = resample_mono(source, 48000, 16000)

        self.assertEqual(output.shape, (1600,))
        self.assertTrue(np.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
