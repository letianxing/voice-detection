import unittest
import numpy as np
from voice_detection.acoustic_reflex import AcousticReflex, LiveBeatDetector, ReflexConfig, config_for_profile
from voice_detection.audio_scene import estimate_tempo, summarize_scores


class ReflexTests(unittest.TestCase):
    def warmed(self):
        reflex = AcousticReflex()
        for stamp in range(0, 3000, 20):
            self.assertEqual(reflex.update(-65, stamp, None, 0), [])
        return reflex

    def test_loud_impulse_without_asr_triggers_once(self):
        reflex = self.warmed()
        # 40 ms of held level, as in the 30-50 ms bursts the startle literature
        # uses; the first two frames alone are not a stimulus.
        self.assertEqual(reflex.update(-8, 3000, None, 0), [])
        self.assertEqual(reflex.update(-8, 3020, None, 0), [])
        events = reflex.update(-8, 3040, None, 0)
        self.assertEqual(events[0]["kind"], "startle")
        self.assertFalse(events[0]["direction_valid"])
        self.assertEqual(reflex.update(-8, 3060, None, 0), [])
        reflex.update(-65, 3080, None, 0)
        for stamp in range(3100, 3200, 20):
            self.assertEqual(reflex.update(-8, stamp, None, 0), [])  # refractory

    def test_a_single_frame_spike_is_a_click_not_a_startle(self):
        reflex = self.warmed()
        for stamp in range(3000, 3400, 20):
            spike = -6 if stamp % 200 == 0 else -65
            self.assertEqual(reflex.update(spike, stamp, None, 0), [], stamp)

    def test_the_bar_follows_how_loud_this_person_actually_is(self):
        close = self.warmed()
        for stamp in range(3000, 4000, 20):  # someone talking right at the mic
            close.update(-12, stamp, None, 0, speech=True)
        quiet = self.warmed()
        for stamp in range(3000, 4000, 20):  # the same person, across the room
            quiet.update(-30, stamp, None, 0, speech=True)
        self.assertAlmostEqual(close.startle_floor(), -3.)   # 抬到设备上限
        self.assertAlmostEqual(quiet.startle_floor(), -8.)   # 设备先验仍然兜底
        bang = []
        for stamp in range(5000, 5100, 20):
            bang += close.update(-8, stamp, None, 0)
        self.assertEqual(bang, [])  # 这个人说话本来就到 -12，-8 不再算惊跳

    def test_music_and_self_echo_do_not_startle(self):
        for kwargs in ({"music": True}, {"echo": .95}):
            self.assertEqual(self.warmed().update(-8, 3000, 150, .95, **kwargs), [])

    def test_orient_requires_sustained_valid_rear_direction(self):
        reflex = self.warmed()
        events = []
        for stamp in range(3000, 3200, 20):
            events += reflex.update(-35, stamp, 140, .95)
        self.assertEqual([event["kind"] for event in events], ["orient"])
        reflex = self.warmed()
        for stamp in range(3000, 3400, 20):
            self.assertEqual(reflex.update(-35, stamp, None, 0), [])


class MusicDSPTests(unittest.TestCase):
    def test_low_genre_scores_remain_unknown(self):
        self.assertFalse(summarize_scores({"Music": .9, "Jazz": .1})["genre_valid"])

    def test_regular_clicks_have_expected_tempo(self):
        rate = 34000
        audio = np.zeros(rate * 12, np.float32)
        click = np.random.default_rng(4).normal(0, .3, 300).astype(np.float32) * np.hanning(300)
        for i in range(4000, len(audio) - len(click), rate // 2):
            audio[i:i + len(click)] = click
        result = estimate_tempo(audio, rate)
        self.assertTrue(result["bpm_valid"], result)
        self.assertAlmostEqual(result["bpm"], 120, delta=4)
        self.assertFalse(estimate_tempo(np.zeros(rate * 8, np.float32), rate)["bpm_valid"])

    def test_beat_pulses_are_observed_not_free_running(self):
        detector = LiveBeatDetector()
        pulses = []
        for index in range(200):
            frame = np.zeros(320, np.float32)
            if index in (25, 50, 75):
                frame[:100] = .3 * np.hanning(100)
            event = detector.update(frame, index * 20, True)
            if event:
                pulses.append(event)
        self.assertEqual(len(pulses), 3)

class BeatPhaseTests(unittest.TestCase):
    def test_offbeat_notes_do_not_double_the_tempo_pulses(self):
        detector = LiveBeatDetector()
        pulses = []
        for index in range(100):
            frame = np.zeros(320, np.float32)
            if index in (12, 25, 37, 50, 62, 75):
                frame[:100] = .3 * np.hanning(100)
            event = detector.update(frame, index * 20, True, bpm=120, beat_anchor_ms=0)
            if event:
                pulses.append(event['stamp_ms'])
        self.assertEqual(pulses, [500, 1000, 1500])

class ConservativeStartleTests(unittest.TestCase):
    def test_normal_voice_after_silence_is_not_a_startle(self):
        reflex=AcousticReflex()
        for stamp in range(0,3000,20):reflex.update(-75,stamp,None,0)
        self.assertEqual(reflex.update(-22,3000,None,0),[])
        self.assertEqual(reflex.update(-22,3020,None,0),[])
        self.assertEqual(reflex.update(-22,3040,None,0),[])
        reflex.update(-75,3060,None,0)
        self.assertEqual(reflex.update(-8,3080,None,0),[])
        self.assertEqual(reflex.update(-8,3100,None,0),[])
        self.assertEqual(reflex.update(-8,3120,None,0)[0]['kind'],'startle')


class DeviceGainTests(unittest.TestCase):
    """The absolute gates follow the level speech reaches on the device in use.

    The Sipeed array records about 18 dB quieter than the built-in microphone, so
    the same dBFS number means a different sound on each. A fixed gate has to be
    wrong on one of them: the -12 dBFS one never fired on the array, and the -45
    that replaced it made ordinary speech on any normal-gain microphone a startle.
    """
    def warmed(self, reflex, floor_dbfs):
        for stamp in range(0, 3000, 20):
            self.assertEqual(reflex.update(floor_dbfs, stamp, None, 0), [])
        return reflex

    def test_speech_on_a_quiet_array_is_not_a_startle_but_a_bang_is(self):
        # 2026-09-16 field measurement: floor -61, loudest speech frame -40.5.
        array = self.warmed(AcousticReflex(ReflexConfig(speech_peak_dbfs=-40.5)), -61)
        for stamp in range(3000, 3100, 20):
            self.assertEqual(array.update(-40.5, stamp, None, 0), [])
        held = []
        for stamp in range(4000, 4100, 20):
            held += array.update(-20, stamp, None, 0)
        self.assertEqual(held[0]['kind'], 'startle')

    def test_the_same_level_is_only_speech_on_a_louder_device(self):
        # -25 dBFS is a bang on the array and an ordinary sentence on this one,
        # which is the whole reason the gate cannot be one number.
        builtin = self.warmed(AcousticReflex(), -61)
        for stamp in range(3000, 3200, 20):
            self.assertEqual(builtin.update(-20, stamp, None, 0), [])  # 阵列上的巨响，这里只是说话
        held = []
        for stamp in range(4000, 4100, 20):
            held += builtin.update(-6, stamp, None, 0)
        self.assertEqual(held[0]['kind'], 'startle')

    def test_profile_carries_the_measured_level_into_both_gates(self):
        from voice_detection.profiles import get_profile
        config = config_for_profile(get_profile('sipeed_6_plus_1_usb_array'))
        self.assertAlmostEqual(config.speech_peak_dbfs, -40.5)
        self.assertAlmostEqual(config.min_startle_dbfs, -26.5)
        self.assertAlmostEqual(config.immediate_startle_dbfs, -24.5)
        mac = config_for_profile(get_profile('mac_builtin'))
        self.assertAlmostEqual(mac.min_startle_dbfs, -8.)
        self.assertAlmostEqual(mac.immediate_startle_dbfs, -6.)
