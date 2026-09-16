import unittest
from voice_detection.dashboard import LiveMonitor
from voice_detection.types import SpeechTranscript


class TranscriptIdentityTests(unittest.TestCase):
    def test_delayed_asr_uses_utterance_speaker_not_current_speaker(self):
        monitor = LiveMonitor()
        monitor.last_speaker = {"speaker_id": "person2", "embedding": [0, 1]}
        monitor.utterance_context[("voice1", 1000)] = {
            "speaker": {"speaker_id": "person1", "similarity": .9, "embedding": [1, 0]},
            "overlap_probability": .1, "self_echo_probability": 0,
        }
        monitor._on_transcript(SpeechTranscript("voice1", "周六去公园", True, started_ms=1000, ended_ms=2000, emitted_ms=3000))
        self.assertEqual(monitor.last_transcript["speaker"]["speaker_id"], "person1")
        self.assertNotIn("embedding", monitor.last_transcript["speaker"])
        self.assertEqual(len(monitor.state()["transcripts"]), 1)

    def test_missing_context_does_not_invent_identity(self):
        monitor = LiveMonitor()
        monitor.last_speaker = {"speaker_id": "person2"}
        monitor._on_transcript(SpeechTranscript("voice1", "一句话", True, started_ms=1000, ended_ms=2000, emitted_ms=3000))
        self.assertEqual(monitor.last_transcript["speaker"], {})

    def test_two_separated_speakers_same_interval_have_distinct_ids(self):
        monitor=LiveMonitor()
        for identity in ['a','b']:
            monitor.utterance_context[('mix',1000)]={'speaker':{'speaker_id':identity,'similarity':.9},'separation':{'verified':True,'speaker_id':identity,'mode':'completed_segment'}}
            monitor._on_transcript(SpeechTranscript('mix','各自说的话',True,started_ms=1000,ended_ms=2000,emitted_ms=3000))
        turns=monitor.state()['transcripts']
        self.assertEqual(len(turns),2)
        self.assertNotEqual(turns[0]['utterance_id'],turns[1]['utterance_id'])
