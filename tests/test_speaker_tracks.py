"""One entry per person in the room, not one per microphone track."""
import unittest

import numpy as np

from voice_detection.speaker_tracks import MixedAudioRing, SpeakerTracker


class FakeEmbedder:
    """Returns whatever vector the caller planted in the audio's first sample."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def embed(self, samples, rate):
        self.calls += 1
        key = int(round(float(samples[0]) * 1000))
        return self.table.get(key)


def speech(key, seconds=1.5, rate=16000):
    """A window long enough to embed: measured on the owner's own enrolment,
    0.6 s windows scored anywhere from 0.108 to 0.708 against his own pool, so
    the tracker now refuses anything under 1.2 s."""
    audio = (np.random.default_rng(key).normal(0, .05, int(rate * seconds))).astype(np.float32)
    audio[0] = key / 1000.0
    return audio


class SpeakerTrackerTest(unittest.TestCase):
    def tracker(self):
        return SpeakerTracker(FakeEmbedder({1: [1.0, 0.0, 0.0, 0.0],
                                            2: [0.0, 1.0, 0.0, 0.0],
                                            3: [0.95, 0.31, 0.0, 0.0]}))

    def test_two_people_taking_turns_become_two_candidates(self):
        tracker = self.tracker()
        tracker.observe(speech(1), 1000, azimuth_deg=-20)
        tracker.observe(speech(2), 1300, azimuth_deg=30)
        tracker.observe(speech(1), 1600, azimuth_deg=-22)

        active = tracker.active(1700)
        self.assertEqual(len(active), 2)
        self.assertEqual({track.speaker_id for track in active}, {"spk_01", "spk_02"})
        first = next(track for track in active if track.speaker_id == "spk_01")
        self.assertEqual(first.samples, 2)
        self.assertLess(first.azimuth_deg, 0)

    def test_the_same_voice_does_not_become_a_second_person(self):
        tracker = self.tracker()
        tracker.observe(speech(1), 1000)
        tracker.observe(speech(3), 1300)   # 0.95 similar: same person, slightly different window

        self.assertEqual(len(tracker.active(1400)), 1)

    def test_somebody_who_stops_talking_drops_out_of_the_conversation(self):
        tracker = self.tracker()
        tracker.observe(speech(1), 1000)
        tracker.observe(speech(2), 1300)

        self.assertEqual(len(tracker.active(1400)), 2)
        self.assertEqual([t.speaker_id for t in tracker.active(4200)], ["spk_02"])

    def test_our_own_playback_is_not_a_speaker(self):
        tracker = self.tracker()

        self.assertIsNone(tracker.observe(speech(1), 1000, echo_probability=.8))
        self.assertEqual(tracker.state(1000)["reason"], "our_own_playback")
        self.assertEqual(tracker.active(1000), [])

    def test_an_enrolled_voice_is_named_rather_than_numbered(self):
        class Profiles:
            def match(self, embedding):
                return ("天行", "owner", .72) if embedding[0] > .9 else ("unknown", "unknown", .1)
        tracker = SpeakerTracker(FakeEmbedder({1: [1.0, 0.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0, 0.0]}),
                                 profiles=Profiles())
        tracker.observe(speech(1), 1000)
        tracker.observe(speech(2), 1300)

        named = {track.speaker_id: track.person_id for track in tracker.active(1400)}
        self.assertEqual(named.get("天行"), "天行")
        self.assertIn("spk_01", named)
        self.assertIsNone(named["spk_01"])


class MixedAudioRingTest(unittest.TestCase):
    """Focused hearing starts after the sentence did; the beginning has to be kept."""

    def test_keeps_the_last_seconds_and_hands_back_a_window(self):
        ring = MixedAudioRing(sample_rate_hz=16000, seconds=2.0)
        for index in range(20):
            ring.append(np.full(1600, index / 100.0, dtype=np.float32), index * 100)

        window, end = ring.window(500)

        self.assertEqual(window.size, 8000)
        self.assertAlmostEqual(float(window[-1]), 0.19, places=3)
        self.assertEqual(ring.span_ms(), 2000)      # older audio is dropped, not grown

    def test_can_look_back_past_the_moment_attention_switched(self):
        ring = MixedAudioRing(sample_rate_hz=16000, seconds=5.0)
        for index in range(30):
            ring.append(np.full(1600, index / 100.0, dtype=np.float32), index * 100)

        recent, _ = ring.window(300)
        earlier, _ = ring.window(300, until_ms=ring.end_ms - 1000)

        self.assertNotAlmostEqual(float(recent[-1]), float(earlier[-1]))
