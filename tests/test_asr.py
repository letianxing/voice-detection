import json
import unittest
import wave
from pathlib import Path

import numpy as np

from voice_detection.asr import AsrScenarioRecord, SherpaSenseVoiceAsrAdapter, resample_mono
from voice_detection.types import SpeechTranscript

SENSEVOICE_DIR = Path(__file__).resolve().parents[1] / "weights" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"


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


class SenseVoiceFinalPassTest(unittest.TestCase):
    """SenseVoice only ever runs on a finished utterance; it cannot stream."""

    def test_missing_model_fails_at_construction_not_mid_utterance(self):
        with self.assertRaises(RuntimeError):
            SherpaSenseVoiceAsrAdapter(Path(__file__).parent / "no-such-model")

    def test_does_not_offer_a_streaming_interface(self):
        # The dashboard picks the streaming recogniser by this attribute; if the
        # offline model ever grew one it would silently take over the partials.
        self.assertFalse(hasattr(SherpaSenseVoiceAsrAdapter, "create_stream"))

    @unittest.skipUnless(SENSEVOICE_DIR.is_dir(), "SenseVoice model not downloaded")
    def test_transcribes_a_finished_chinese_utterance(self):
        adapter = SherpaSenseVoiceAsrAdapter(SENSEVOICE_DIR)
        with wave.open(str(SENSEVOICE_DIR / "test_wavs" / "zh.wav")) as handle:
            rate = handle.getframerate()
            audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

        self.assertIsNone(adapter.transcribe("voice_1", audio[:160], rate))  # 10 ms is not an utterance
        result = adapter.transcribe("voice_1", audio, rate)

        self.assertTrue(result.is_final)
        self.assertEqual(result.track_id, "voice_1")
        self.assertIn("开饭时间", result.text)
        # Punctuation and inverse text normalisation are what this pass adds over
        # the streaming model's "早上九点至下午五点"; both reach memory and the LLM.
        self.assertIn("9点", result.text)


if __name__ == "__main__":
    unittest.main()
