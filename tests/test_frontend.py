import unittest

import numpy as np

from voice_detection.frontend import CocktailFrontend
from voice_detection.profiles import get_profile, load_profiles
from voice_detection.ros4hri import acoustic_track_to_attention_json, acoustic_track_to_ros4hri_json
from voice_detection.simulation import synthetic_array_frame
from voice_detection.types import AudioFrame


class VoiceFrontendTest(unittest.TestCase):
    def test_profiles_include_required_devices(self):
        profiles = load_profiles()

        self.assertIn("mac_builtin", profiles)
        self.assertIn("sipeed_6_plus_1_usb_array", profiles)
        self.assertTrue(profiles["sipeed_6_plus_1_usb_array"].supports_doa)

    def test_mac_builtin_produces_mono_track_without_doa(self):
        profile = get_profile("mac_builtin")
        t = np.arange(4800, dtype=np.float32) / float(profile.sample_rate_hz)
        frame = AudioFrame(
            samples=(np.sin(2.0 * np.pi * 220.0 * t).reshape((-1, 1)) * 0.08).astype(np.float32),
            sample_rate_hz=profile.sample_rate_hz,
            stamp_ms=1000,
        )

        output = CocktailFrontend(profile).process(frame)

        self.assertEqual(len(output.tracks), 1)
        self.assertEqual(output.tracks[0].track_id, "voice_mono")
        self.assertIsNone(output.tracks[0].azimuth_deg)
        self.assertTrue(output.tracks[0].voice_activity)

    def test_sipeed_array_estimates_azimuth_and_outputs_target_audio(self):
        profile = get_profile("sipeed_6_plus_1_usb_array")
        frame = synthetic_array_frame(profile, azimuth_deg=30.0)

        output = CocktailFrontend(profile).process(frame)

        self.assertEqual(len(output.tracks), 1)
        self.assertIsNotNone(output.tracks[0].azimuth_deg)
        self.assertLess(abs(output.tracks[0].azimuth_deg - 30.0), 16.0)
        self.assertEqual(output.target_audio.ndim, 1)
        self.assertEqual(output.target_audio.shape[0], frame.samples.shape[0])

    def test_ros4hri_mapping_keeps_engineering_track(self):
        profile = get_profile("sipeed_6_plus_1_usb_array")
        output = CocktailFrontend(profile).process(synthetic_array_frame(profile, azimuth_deg=-45.0))
        track = output.tracks[0]

        ros4hri = acoustic_track_to_ros4hri_json(track)
        attention = acoustic_track_to_attention_json(track)

        self.assertEqual(ros4hri["tracked_topic"], "/humans/voices/tracked")
        self.assertIn("/humans/voices/", ros4hri["voice_topics"]["speech"])
        self.assertEqual(attention["track_id"], track.track_id)
        self.assertIn("clarity", attention)


if __name__ == "__main__":
    unittest.main()
