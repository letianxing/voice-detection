import unittest
from voice_detection.echo_guard import match_playback


class EchoTests(unittest.TestCase):
    def setUp(self):
        self.records=[{'id':'p','turn_id':'t','text':'我们继续聊之前公园的话题。','started_ms':1000,'ended_ms':3000}]

    def test_contemporaneous_robot_phrase_is_echo(self):
        self.assertEqual(match_playback('继续聊之前公园的话题',1200,3100,self.records)['turn_id'],'t')

    def test_later_repetition_and_independent_speaker_are_not_echo(self):
        self.assertIsNone(match_playback('继续聊之前公园的话题',5000,6000,self.records))
        self.assertIsNone(match_playback('继续聊之前公园的话题',1200,3100,self.records,.9))
        self.assertIsNone(match_playback('小艾克斯，停一下，我有问题',1200,3000,self.records))
