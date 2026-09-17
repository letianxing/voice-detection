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
