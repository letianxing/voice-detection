import unittest
import numpy as np
from voice_detection.acoustic_reflex import AcousticReflex, LiveBeatDetector
from voice_detection.audio_scene import estimate_tempo, summarize_scores


class ReflexTests(unittest.TestCase):
    def warmed(self):
        reflex = AcousticReflex()
        for stamp in range(0, 3000, 20):
            self.assertEqual(reflex.update(-65, stamp, None, 0), [])
        return reflex

    def test_loud_impulse_without_asr_triggers_once(self):
        reflex = self.warmed()
        events = reflex.update(-8, 3000, None, 0)
        self.assertEqual(events[0]["kind"], "startle")
        self.assertFalse(events[0]["direction_valid"])
        self.assertEqual(reflex.update(-8, 3020, None, 0), [])
        reflex.update(-65, 3040, None, 0)
        self.assertEqual(reflex.update(-8, 3060, None, 0), [])

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
        reflex.update(-75,3020,None,0)
        self.assertEqual(reflex.update(-8,3040,None,0)[0]['kind'],'startle')
