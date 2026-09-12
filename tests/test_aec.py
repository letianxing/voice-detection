import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from voice_detection.aec import JsonlPlaybackReferenceSource, LinearEchoCanceller, PlaybackReferenceBuffer
from voice_detection.audio_frame_codec import encode_audio_frame_payload
from voice_detection.types import AudioFrame


class AecTest(unittest.TestCase):
    def test_linear_echo_canceller_reduces_aligned_reference(self):
        reference = np.sin(np.linspace(0.0, 16.0, 960, dtype=np.float32)) * 0.2
        microphone = reference * 0.8
        canceller = LinearEchoCanceller(smoothing=0.0)

        result = canceller.process(microphone, reference)

        self.assertLess(float(np.mean(result**2)), float(np.mean(microphone**2)) * 0.01)

    def test_reference_buffer_aligns_and_resamples(self):
        buffer = PlaybackReferenceBuffer()
        reference = AudioFrame(np.ones((160, 1), dtype=np.float32), 16000, 1010)
        capture = AudioFrame(np.zeros((960, 1), dtype=np.float32), 48000, 1000)
        buffer.append(reference)

        aligned = buffer.read_for(capture)

        self.assertTrue(np.allclose(aligned[:480], 0.0))
        self.assertTrue(np.allclose(aligned[480:], 1.0))

    def test_jsonl_source_tails_audio_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"
            source = JsonlPlaybackReferenceSource(path, poll_interval_sec=0.002)
            try:
                payload = encode_audio_frame_payload(
                    np.ones(160, dtype=np.float32), 16000, "tts", stamp_sec=1
                ).decode("utf-8")
                path.write_text(payload + "\n", encoding="utf-8")
                deadline = time.monotonic() + 0.5
                while source.offset == 0 and time.monotonic() < deadline:
                    time.sleep(0.005)
                capture = AudioFrame(np.zeros((160, 1), dtype=np.float32), 16000, 1000)
                self.assertTrue(np.allclose(source.read_for(capture), 1.0))
            finally:
                source.close()

    def test_jsonl_source_ignores_existing_session_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"
            payload = encode_audio_frame_payload(
                np.ones(160, dtype=np.float32), 16000, "old", stamp_sec=1
            ).decode("utf-8")
            path.write_text(payload + "\n", encoding="utf-8")
            source = JsonlPlaybackReferenceSource(path, poll_interval_sec=0.002)
            try:
                time.sleep(0.02)
                capture = AudioFrame(np.zeros((160, 1), dtype=np.float32), 16000, 1000)
                self.assertTrue(np.allclose(source.read_for(capture), 0.0))
            finally:
                source.close()


if __name__ == "__main__":
    unittest.main()
