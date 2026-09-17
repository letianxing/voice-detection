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
        self.assertIsNone(match_playback('小圆，停一下，我有问题',1200,3000,self.records))


class MisheardEchoTests(unittest.TestCase):
    """机器人听见自己、又回答自己，一轮变三轮。

    The field sequence, from the service's own transcript log:

      17:55:29 天行「你叫什么名字？」        self_echo 0.000
      17:55:31 robot plays「我是小圆呀」
      17:55:32 heard 「我是小圆样。」        self_echo 0.457, text score 0.80
      17:55:33 robot answers *that*「我是小圆呀」 → and so on.

    One character wrong under the robot's own voice was enough to get under the
    text-only bar of 0.82. The residual was the evidence the text lacked: every
    real person in that session came out at exactly 0.000 and every echo
    between 0.457 and 0.758.
    """

    def reply(self, text, start, end):
        return [{'id': 'p', 'turn_id': 't', 'text': text, 'started_ms': start, 'ended_ms': end}]

    def test_one_character_misheard_is_still_our_own_voice(self):
        match = match_playback('我是小圆样。', 1200, 2200, self.reply('我是小圆呀', 1000, 2400),
                               .49, .457)
        self.assertIsNotNone(match)
        self.assertEqual(match['reason'], 'echo_residual_over_our_own_playback')
        self.assertEqual(match['echo_residual'], .457)

    def test_a_mangled_tail_is_still_our_own_voice(self):
        match = match_playback('上海今天阴天27度，9点人0。', 1200, 3800,
                               self.reply('上海今天阴天，27度，有点闷。', 1000, 4000), .47, .689)
        self.assertIsNotNone(match)

    def test_a_short_fragment_of_what_we_just_said_is_our_own_voice(self):
        match = match_playback('上海。', 1200, 1700,
                               self.reply('上海今天阴天，27度，有点闷。', 1000, 4000), .40, .501)
        self.assertIsNotNone(match)

    def test_but_a_short_human_answer_is_not(self):
        self.assertIsNone(match_playback('好的', 1200, 1700,
                                         self.reply('那我们明天再聊好的地方', 1000, 4000), .4, 0.))

    def test_and_somebody_talking_over_us_is_not(self):
        """Barge-in: different words, and the residual stays low because what is
        left after cancellation is the person, not us."""
        self.assertIsNone(match_playback('等一下我有个问题', 1200, 2400,
                                         self.reply('上海今天阴天，27度，有点闷。', 1000, 4000), .4, .38))

    def test_nor_is_a_person_saying_something_similar_when_nothing_played(self):
        self.assertIsNone(match_playback('我是小圆样。', 6000, 7000,
                                         self.reply('我是小圆呀', 1000, 2400), .49, .457))
