import unittest
import tempfile
from pathlib import Path
from voice_detection.speaker_embedding import SpeakerProfileStore

class StrangerTests(unittest.TestCase):
    def test_unknown_voices_persist_and_registration_keeps_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'voices.json';store=SpeakerProfileStore(path)
            a,role,_=store.remember_stranger([1.,0.,0.]);b,_,_=store.remember_stranger([0.,1.,0.])
            self.assertEqual(role,'stranger');self.assertNotEqual(a,b)
            store=SpeakerProfileStore(path)
            self.assertEqual(store.match([1.,0.,0.])[0],a)
            store.enroll('小天',[1.,0.,0.],'owner')
            self.assertEqual(store.match([1.,0.,0.])[:2],('小天','owner'))
            self.assertIn(a,store.profiles['小天']['aliases'])

    def test_registration_rejects_inconsistent_voice_segments(self):
        import numpy as np
        from unittest.mock import Mock
        from voice_detection.speaker_embedding import SpeakerEmbedder
        embedder=SpeakerEmbedder.__new__(SpeakerEmbedder)
        embedder.embed=Mock(side_effect=[[1.,0.],[0.,1.]])
        with self.assertRaises(ValueError):embedder.enrollment_embedding(np.zeros(32000),16000)
        embedder.embed=Mock(return_value=[1.,0.])
        self.assertEqual(embedder.enrollment_embedding(np.zeros(32000),16000),[1.,0.])

    def test_pool_retains_anchor_rejects_drift_and_requires_independent_face(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            store=SpeakerProfileStore(Path(tmp)/'pool.json')
            store.enroll('a',[1,0,0],'owner')
            for angle in np.linspace(.2,.7,10):store.enroll('a',[np.cos(angle),np.sin(angle),0],'owner')
            self.assertLessEqual(len(store.profiles['a']['embeddings']),8)
            self.assertEqual(store.profiles['a']['embedding'],[1,0,0])
            self.assertEqual(store.match([1,0,0])[0],'a')
            quality={'echo':0,'overlap':0,'duration_ms':3000}
            self.assertFalse(store.add_verified_sample('a',[1,0,0],'b',.99,quality))
            with self.assertRaises(ValueError):store.enroll('a',[0,0,1],'owner')
            restored=SpeakerProfileStore(Path(tmp)/'pool.json');self.assertEqual(restored.match([1,0,0])[0],'a')


class VoicePoolLearningTest(unittest.TestCase):
    """The owner leaning back in a chair has to become part of the owner's pool.

    Field trace: enrolled face-on at close range, then measured 0.0 off-axis at
    -90 degrees, filed as stranger_9a227aac8372, answer dropped. The pool could
    not learn the new angle because learning required a .7 match against the very
    pool that was missing it.
    """

    def store(self, path):
        from voice_detection.speaker_embedding import SpeakerProfileStore
        store = SpeakerProfileStore(str(path))
        store.enroll("天行", [1.0, 0.0, 0.0, 0.0], "owner")
        return store

    def good(self):
        return {"overlap": 0.0, "echo": 0.0, "duration_ms": 3000}

    def test_a_confident_face_teaches_the_pool_an_unfamiliar_angle(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as folder:
            store = self.store(pathlib.Path(folder) / "p.json")
            off_axis = [0.40, 0.9165, 0.0, 0.0]   # ~0.40 against the anchor: same person, other angle
            self.assertEqual(store.match(off_axis)[0], "unknown")   # the voiceprint alone cannot tell

            self.assertTrue(store.add_verified_sample("天行", off_axis, "天行", .9, self.good()))
            self.assertEqual(store.match(off_axis)[0], "天行")      # and now it can

    def test_a_face_cannot_put_a_different_enrolled_voice_into_the_pool(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as folder:
            store = self.store(pathlib.Path(folder) / "p.json")
            store.enroll("二丫", [0.0, 1.0, 0.0, 0.0], "known")

            self.assertFalse(store.add_verified_sample("天行", [0.05, 0.99, 0.0, 0.0], "天行", .95, self.good()))

    def test_overlapping_or_echoing_speech_still_teaches_nothing(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as folder:
            store = self.store(pathlib.Path(folder) / "p.json")
            sample = [0.9, 0.44, 0.0, 0.0]
            self.assertFalse(store.add_verified_sample("天行", sample, "天行", .9, {"overlap": .5, "echo": 0., "duration_ms": 3000}))
            self.assertFalse(store.add_verified_sample("天行", sample, "天行", .9, {"overlap": 0., "echo": .5, "duration_ms": 3000}))
            self.assertFalse(store.add_verified_sample("天行", sample, "天行", .5, self.good()))

    def test_stranger_clusters_do_not_block_the_owner(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as folder:
            store = self.store(pathlib.Path(folder) / "p.json")
            # what the field session accumulated: the same person, filed nine times
            for index in range(4):
                store.enroll(f"stranger_{index:04d}", [0.9 - index * 0.01, 0.44, 0.0, 0.0], "stranger")

            speaker, role, score = store.match([0.93, 0.37, 0.0, 0.0])

            self.assertEqual(speaker, "天行")
            self.assertEqual(role, "owner")


class ConfidenceTests(unittest.TestCase):
    """How the decision is made, not just what it is."""

    def store(self, tmp):
        """Ten voices with nothing in common, so there is a cohort to stand above.

        Deliberately orthogonal: the point of the test is the decision rule, and
        a cohort built from random vectors would make the z score depend on the
        seed rather than on the rule.
        """
        import numpy as np
        store = SpeakerProfileStore(Path(tmp) / 'norm.json')
        for index in range(1, 11):
            store.enroll(f'other{index}', np.eye(32)[index].tolist(), 'stranger')
        return store

    def test_a_score_far_above_the_other_voices_names_somebody(self):
        """AS-norm: the absolute cosine is not the question.

        Measured on this machine, every ERes2NetV2 embedding shares a large
        common component — mean cosine 0.38 between clips of different people
        — so a fixed 0.48 bar is a bar on the channel as much as on the voice.
        """
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store(tmp)
            anchor = np.eye(32)[0]
            store.enroll('天行', anchor.tolist(), 'owner')
            # 0.42 of his own voice and the rest in a direction nobody else
            # occupies: the channel moved, the person did not.
            shifted = anchor * .42 + np.eye(32)[20] * (1 - .42 ** 2) ** .5
            decision = store.identify(shifted.tolist())
            self.assertEqual(decision['speaker_id'], '天行')
            self.assertEqual(decision['path'], 'cohort_normalized')
            self.assertLess(decision['score'], store.thresholds['threshold'])
            self.assertGreaterEqual(decision['z'], store.thresholds['z_floor'])

    def test_nothing_is_named_from_noise_however_it_normalises(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store(tmp)
            anchor = np.eye(32)[0]
            store.enroll('天行', anchor.tolist(), 'owner')
            # A quarter of his voice and three quarters of the room: the best
            # candidate still stands above the cohort, and that is not enough.
            faint = anchor * .25 + np.eye(32)[21] * (1 - .25 ** 2) ** .5
            decision = store.identify(faint.tolist())
            self.assertEqual(decision['speaker_id'], 'unknown')
            self.assertIn(decision['reason'], {'below_absolute_and_cohort_bars',
                                               'another_voice_scores_almost_the_same'})

    def test_thresholds_are_tunable_at_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(Path(tmp) / 'tune.json')
            self.assertEqual(store.tune(threshold=.4, z_floor=1.5)['threshold'], .4)
            self.assertEqual(store.thresholds['z_floor'], 1.5)
            self.assertNotIn('nonsense', store.tune(nonsense=1))

    def test_auto_registered_voices_are_numbered_and_capped(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(Path(tmp) / 'crowd.json')
            rng = np.random.default_rng(11)
            names = []
            for _ in range(20):
                identity, _, _ = store.remember_stranger(rng.normal(0, 1, 32).tolist())
                if identity.startswith('stranger_'):
                    names.append(store.display_name(identity))
            self.assertIn('陌生人1', names)
            strangers = [key for key, value in store.profiles.items() if value.get('role') == 'stranger']
            self.assertLessEqual(len(strangers), store.MAX_STRANGERS)

    def test_a_cluster_the_face_identified_ends_up_in_that_persons_pool(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(Path(tmp) / 'merge.json')
            anchor = np.eye(32)[0]
            store.enroll('天行', anchor.tolist(), 'owner')
            # Close enough to be him at another distance, not close enough for
            # the voiceprint to say so on its own — the case that fills a store
            # up with clusters of people it already knows.
            nearby = anchor * .40 + np.eye(32)[1] * (1 - .40 ** 2) ** .5
            cluster, role, _ = store.remember_stranger(nearby.tolist())
            self.assertEqual(role, 'stranger')
            self.assertTrue(store.merge_cluster(cluster, '天行'))
            self.assertEqual(store.profiles[cluster], {'canonical_id': '天行', 'role': 'alias',
                                                       'merged_samples': 1})
            self.assertIn(cluster, store.profiles['天行']['aliases'])
            self.assertEqual(store.display_name(cluster), '天行')

    def test_two_agreeing_face_verified_samples_open_a_second_anchor(self):
        """One confident face is not enough to let a voice into an identity.

        The owner's enrolment is a single quiet clip; his live sentences scored
        0.083 against it. Two samples that a confident face named and that also
        agree with each other say the same person through a different channel.
        """
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(Path(tmp) / 'anchor.json')
            anchor = np.zeros(32); anchor[0] = 1.
            store.enroll('天行', anchor.tolist(), 'owner')
            quality = {'echo': 0., 'overlap': 0., 'duration_ms': 3000}
            far = np.eye(32)[5] * .9 + np.eye(32)[6] * .43
            self.assertFalse(store.add_verified_sample('天行', far.tolist(), '天行', .9, quality))
            self.assertEqual(store.profiles['天行']['anchor_groups'], 1)
            again = far * .9 + np.eye(32)[7] * .2
            self.assertTrue(store.add_verified_sample('天行', again.tolist(), '天行', .9, quality))
            self.assertEqual(store.profiles['天行']['anchor_groups'], 2)
            self.assertEqual(store.match(far.tolist())[0], '天行')
            self.assertEqual(store.profiles['天行']['embedding'], anchor.tolist())

    def test_a_single_face_verified_sample_never_joins_a_pool_by_itself(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(Path(tmp) / 'lonely.json')
            anchor = np.zeros(32); anchor[0] = 1.
            store.enroll('天行', anchor.tolist(), 'owner')
            quality = {'echo': 0., 'overlap': 0., 'duration_ms': 3000}
            rng = np.random.default_rng(5)
            for _ in range(3):
                store.add_verified_sample('天行', rng.normal(0, 1, 32).tolist(), '天行', .9, quality)
            self.assertEqual(store.profiles['天行']['anchor_groups'], 1)
