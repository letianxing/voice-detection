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
