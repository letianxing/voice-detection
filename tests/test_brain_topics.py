import json
from pathlib import Path
import tempfile
import unittest
from voice_detection.brain_topics import BrainTopicMapper, SpeakerIDs, ROOT


class TopicTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ids = SpeakerIDs(Path(self.temp.name) / "ids.json")
        self.mapper = BrainTopicMapper(ids=self.ids)

    def state(self, angle=None):
        return {"running": True, "audio_scene": {"acoustic": {"stamp_ms": 1000, "direction_valid": angle is not None, "direction_deg": angle},
                "music": {"stamp_ms": 1000, "valid": True, "music_state": True, "bpm_valid": True, "bpm": 120, "genre_valid": True, "genre": "jazz"},
                "events": [{"id": "e", "kind": "startle", "stamp_ms": 1000}]},
                "transcripts": [{"utterance_id": "u", "track_id": "v1", "is_final": True, "text": "你好", "emitted_ms": 1000,
                                 "direction": {"direction_valid": angle is not None, "direction_deg": angle},
                                 "speaker": {"speaker_id": "person1", "speaker_role": "owner"}, "speaker_vector": [.125, -.375]}]}

    def test_unknown_angle_not_published_and_floats_preserved(self):
        attention = {"utterances": [{"utterance_id": "u", "attention": {"listen": True, "addressed_to_robot": True, "confidence": .9}}]}
        messages = dict(self.mapper.messages(self.state(), 1100, attention))
        self.assertNotIn("/sound_direction", messages)
        voice = json.loads(messages["/voice_msg"]["data"])
        self.assertNotIn("angle", voice)
        self.assertFalse(voice["angle_valid"])
        self.assertEqual(voice["user_id"], "1")
        self.assertEqual(voice["speaker_label"], "person1")
        self.assertEqual(voice["speaker_vector"], [.125, -.375])
        self.assertEqual(SpeakerIDs(self.ids.path).numeric("person1"), "1")

    def test_bystander_asr_not_sent_to_production_brain(self):
        messages = dict(self.mapper.messages(self.state(), 1100, {}))
        self.assertNotIn("/voice_msg", messages)
        self.assertIn("/voice/perception_state", messages)

    def test_front_angle_converts_to_production_90_and_events_deduplicate(self):
        messages = dict(self.mapper.messages(self.state(0), 1100))
        self.assertEqual(messages["/sound_direction"]["data"], 90)
        self.assertEqual(messages["/startle_trigger"]["data"], True)
        self.assertNotIn("/startle_trigger", dict(self.mapper.messages(self.state(0), 1150)))
        expired = dict(self.mapper.messages(self.state(0), 10000))
        self.assertNotIn("/sound_direction", expired)
        self.assertFalse(expired["/audio/music_state"]["data"])
        self.assertEqual(expired["/music_bpm"]["data"], 0)
        self.assertEqual(expired["/music_genre"]["data"], "unknown")

    def test_typed_contract_withholds_unknown_direction(self):
        self.mapper.config.update(voice_topic="/voice_msg_speaker", voice_type="audio_msgs/msg/AudioMsg", require_attention=False)
        self.assertNotIn("/voice_msg_speaker", dict(self.mapper.messages(self.state(), 1100)))
        self.assertIn("/voice_msg_speaker", dict(self.mapper.messages(self.state(0), 1150)))

class IntentTests(unittest.TestCase):
    def test_only_exact_commands_are_explicit_intents(self):
        from voice_detection.brain_topics import explicit_intent
        self.assertEqual(explicit_intent('小艾克斯，跟着我'), 'follow')
        self.assertEqual(explicit_intent('停止跟随'), 'stop')
        self.assertIsNone(explicit_intent('他说让你跟着我，你觉得呢？'))
