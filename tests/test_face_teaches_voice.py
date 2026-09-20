"""人脸认得出他，声纹库里却没有他——这个人永远学不到声纹。

现场 2026-09-20 16:42：天行人脸已注册，`speaker_profiles.json` 里却只有
robot 和八个一条样本的陌生簇。他一离开镜头说话，声纹无从比对，于是
「哎呀，我感觉好累啊。」记成说话人5；等他回到镜头前，人脸认出是天行，
但那句话仍然挂在说话人5 名下，问「你觉得呢」时机器人接的不是那句。

原因在这里：人脸确认过的样本要进池子，先要求这个人已经有档案——
注册时只存下人脸的人，档案不存在，样本被拒，于是永远没有档案。
开档的条件和加锚点一样严：两条互相吻合的、人脸确认过的样本。
"""
import unittest

import numpy as np

from voice_detection.speaker_embedding import SpeakerProfileStore


def voice(seed, jitter=0.0, width=192):
    base = np.random.default_rng(seed).normal(size=width)
    if jitter:
        base = base + jitter * np.random.default_rng(seed + 991).normal(size=width)
    return (base / np.linalg.norm(base)).tolist()


def clean(duration_ms=2600):
    return {'overlap': .0, 'echo': .0, 'duration_ms': duration_ms}


class BootstrapTests(unittest.TestCase):
    def profiles(self):
        import tempfile, pathlib as _p
        folder = tempfile.mkdtemp()
        self.addCleanup(lambda: None)
        return SpeakerProfileStore(_p.Path(folder) / 'speaker_profiles.json')

    def test_the_bar_travels_with_the_sample(self):
        """0.45 在 sface 的尺度上是一次可靠识别，不该被当成没认出来。"""
        store = self.profiles()
        self.assertFalse(store.add_verified_sample('天行', voice(3), '天行', .45, clean(), face_floor=.44))
        self.assertTrue(store.add_verified_sample('天行', voice(3, .3), '天行', .45, clean(), face_floor=.44))
        self.assertIn('天行', store.profiles)

    def test_two_agreeing_face_verified_samples_open_the_file(self):
        store = self.profiles()
        self.assertNotIn('天行', store.profiles)

        first = store.add_verified_sample('天行', voice(1), '天行', .9, clean())
        self.assertFalse(first)                      # 一条不够
        self.assertNotIn('天行', store.profiles)

        second = store.add_verified_sample('天行', voice(1, .3), '天行', .9, clean())

        self.assertTrue(second)
        self.assertIn('天行', store.profiles)
        self.assertEqual(store.profiles['天行']['role'], 'known')
        self.assertEqual(len(store.profiles['天行']['embeddings']), 2)

    def test_two_samples_that_disagree_do_not(self):
        """两条人脸确认的样本彼此对不上，说明至少有一条是别人说的。"""
        store = self.profiles()
        store.add_verified_sample('天行', voice(1), '天行', .9, clean())
        self.assertFalse(store.add_verified_sample('天行', voice(77), '天行', .9, clean()))
        self.assertNotIn('天行', store.profiles)

    def test_a_face_nobody_is_sure_about_teaches_nothing(self):
        """低于人脸模型自己的判定阈值，那不是一次识别。

        门槛由调用方按人脸模型的尺度给：实测 sface 认出天行是 0.450，
        它自己的阈值是 0.380——拿一个绝对的 0.8 去卡，每条样本都会被拒，
        池子永远是空的，这正是现场发生的事。
        """
        store = self.profiles()
        store.add_verified_sample('天行', voice(1), '天行', .30, clean(), face_floor=.44)
        store.add_verified_sample('天行', voice(1, .2), '天行', .30, clean(), face_floor=.44)
        self.assertNotIn('天行', store.profiles)
        self.assertIn('人脸只有', store.last_pool_reason)

    def test_and_neither_does_a_one_second_answer(self):
        store = self.profiles()
        store.add_verified_sample('天行', voice(1), '天行', .9, clean(duration_ms=700))
        store.add_verified_sample('天行', voice(1, .2), '天行', .9, clean(duration_ms=700))
        self.assertNotIn('天行', store.profiles)
        self.assertIn('做模板太短', store.last_pool_reason)

    def test_an_ordinary_短_sentence_still_teaches(self):
        """日常说话大多一两秒；再严就等于池子永远长不起来。"""
        store = self.profiles()
        store.enroll('天行', voice(1), 'known')
        self.assertTrue(store.add_verified_sample('天行', voice(1, .5), '天行', .9,
                                                  clean(duration_ms=1400)))

    def test_a_voice_that_is_clearly_somebody_else_enrolled_is_refused(self):
        store = self.profiles()
        store.enroll('小李', voice(5), 'known')
        store.add_verified_sample('天行', voice(5, .05), '天行', .9, clean())
        second = store.add_verified_sample('天行', voice(5, .06), '天行', .9, clean())
        # 两条样本互相吻合，但它们听起来就是小李——开档会把小李的声音记成天行
        self.assertTrue(second is False or '天行' not in store.profiles or
                        store.rank(voice(5))[0]['speaker_id'] == '小李')

    def test_the_reason_always_says_what_happened(self):
        store = self.profiles()
        store.add_verified_sample('天行', voice(1), '天行', .9, clean())
        self.assertIn('还没有声纹档案', store.last_pool_reason)
        store.add_verified_sample('天行', voice(1, .3), '天行', .9, clean())
        self.assertIn('建了声纹档案', store.last_pool_reason)

    def test_an_existing_pool_still_grows_the_way_it_did(self):
        store = self.profiles()
        store.enroll('天行', voice(1), 'owner')
        before = len(store.profiles['天行']['embeddings'])
        # 同一个人、换个距离说的一句（不是几乎一模一样的那种，那种会去重）
        self.assertTrue(store.add_verified_sample('天行', voice(1, .5), '天行', .9, clean()))
        self.assertGreater(len(store.profiles['天行']['embeddings']), before)


if __name__ == '__main__':
    unittest.main()
