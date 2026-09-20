"""机器人自己说的话，不能和用户说的话并成一句。

现场 2026-09-20：机器人播「上海晴，28度。」，人接着说「我让你查冰淇淋店。」，
中间没有够 480 毫秒的静音，于是两个人的声音进了同一段——四秒的音频、两个
说话人、一条转写。往下走没有一个好结果：回声判别给了 0.275 自身回声（整段
大部分是机器人，于是被当成自己的声音丢掉），人只好把那句话重说一遍；反过来
如果人说的那半更长，机器人就会去回答自己刚说过的话。
"""
import unittest

import numpy as np

from voice_detection.streaming_asr import UtteranceSegmenter


class PlaybackBoundaryTests(unittest.TestCase):
    RATE = 16000

    def chunk(self, ms=100, loud=True):
        samples = np.full(int(self.RATE * ms / 1000), .2 if loud else 0., dtype=np.float32)
        return samples

    def feed(self, segmenter, count, stamp, speech=True, playing=False, step=100):
        """Returns every utterance the segmenter emitted while feeding."""
        emitted = []
        for index in range(count):
            out = segmenter.process(self.chunk(step), self.RATE, stamp + index * step,
                                    speech, "voice_mono", playing=playing)
            if out is not None:
                emitted.append(out)
        return emitted, stamp + count * step

    def test_the_robots_own_audio_and_the_answer_are_two_utterances(self):
        segmenter = UtteranceSegmenter(playback_tail_ms=0)
        played, stamp = self.feed(segmenter, 20, 100000, speech=True, playing=True)   # 2 s 机器人在播
        self.assertEqual(played, [])
        answered, stamp = self.feed(segmenter, 12, stamp, speech=True, playing=False)  # 人紧接着说

        # 播放停的那一刻就切了一段：机器人那段单独成句。
        self.assertEqual(len(answered), 1)
        robot_part = answered[0]
        self.assertAlmostEqual(robot_part.samples.size / self.RATE, 2.0, delta=.25)
        # 人说的那段还在进行中，没有被并进去。
        self.assertTrue(segmenter.active)
        tail = segmenter.flush(self.RATE, stamp)
        self.assertIsNotNone(tail)
        self.assertAlmostEqual(tail.samples.size / self.RATE, 1.2, delta=.25)

    def test_the_cut_does_not_carry_the_robots_tail_into_the_next_sentence(self):
        """切开时把 pre-roll 丢掉：那 180 毫秒是对面的声音。"""
        segmenter = UtteranceSegmenter(playback_tail_ms=0)
        _, stamp = self.feed(segmenter, 10, 100000, speech=True, playing=True)
        _, stamp = self.feed(segmenter, 6, stamp, speech=True, playing=False)
        human = segmenter.flush(self.RATE, stamp)
        self.assertAlmostEqual(human.samples.size / self.RATE, .6, delta=.15)

    def test_the_speaker_still_ringing_belongs_to_the_robot(self):
        """播放刚停的那一小段仍算机器人的：音箱的余音不该带进下一句。"""
        segmenter = UtteranceSegmenter(playback_tail_ms=300)
        _, stamp = self.feed(segmenter, 10, 100000, speech=True, playing=True)
        emitted, stamp = self.feed(segmenter, 2, stamp, speech=True, playing=False)   # 200 ms 余音
        self.assertEqual(emitted, [])                                                 # 还没切
        emitted, stamp = self.feed(segmenter, 4, stamp, speech=True, playing=False)
        self.assertEqual(len(emitted), 1)
        self.assertGreater(emitted[0].samples.size / self.RATE, 1.1)                  # 余音留在机器人这段

    def test_a_sentence_nobody_interrupted_is_still_one_utterance(self):
        """没有播放事件时，行为和以前完全一样。"""
        segmenter = UtteranceSegmenter(playback_tail_ms=0)
        emitted, stamp = self.feed(segmenter, 15, 100000, speech=True, playing=False)
        self.assertEqual(emitted, [])
        emitted, stamp = self.feed(segmenter, 6, stamp, speech=False, playing=False)  # 600 ms 静音收句
        self.assertEqual(len(emitted), 1)
        # 1.5 秒说话 + 收句前的静音，一段，和以前一样
        self.assertAlmostEqual(emitted[0].samples.size / self.RATE, 2.0, delta=.3)

    def test_the_robot_starting_to_speak_also_closes_what_was_being_said(self):
        segmenter = UtteranceSegmenter(playback_tail_ms=0)
        _, stamp = self.feed(segmenter, 8, 100000, speech=True, playing=False)
        emitted, stamp = self.feed(segmenter, 4, stamp, speech=True, playing=True)
        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(emitted[0].samples.size / self.RATE, .8, delta=.2)


if __name__ == '__main__':
    unittest.main()
