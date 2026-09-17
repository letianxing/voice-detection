import unittest
import tempfile
import numpy as np
from pathlib import Path
from voice_detection.speaker_embedding import SpeakerProfileStore

class StrangerTests(unittest.TestCase):
    @staticmethod
    def strangers(store):
        return [k for k,v in store.profiles.items() if v.get('role')=='stranger']

    @staticmethod
    def at(similarity_to_x):
        t=float(np.arccos(similarity_to_x));return [float(np.cos(t)),float(np.sin(t)),0.]

    def test_one_utterance_opens_a_provisional_profile_that_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=SpeakerProfileStore(Path(tmp)/'voices.json')
            self.assertEqual(store.remember_stranger([1.,0.,0.],duration_ms=5000,now_ms=1000)[:2],('unknown','unknown'))
            ids=self.strangers(store)
            self.assertEqual(len(ids),1);self.assertEqual(store.profiles[ids[0]]['lifecycle'],'provisional')
            self.assertEqual(store.match([1.,0.,0.])[0],'unknown')      # not published
            self.assertEqual(store._match_any([1.,0.,0.])[0],ids[0])   # but matched internally

    def test_three_long_utterances_promote_and_publish_the_stranger(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=SpeakerProfileStore(Path(tmp)/'voices.json')
            # three sentences of one voice: ≈.85 to the first, spread over different axes like real speech
            s1=[1.,0.,0.];s2=[.85,float(np.sqrt(1-.85**2)),0.];s3=[.85,0.,float(np.sqrt(1-.85**2))]
            self.assertEqual(store.remember_stranger(s1,duration_ms=5000,now_ms=1000)[0],'unknown')
            self.assertEqual(store.remember_stranger(s2,duration_ms=5000,now_ms=2000)[0],'unknown')
            sid,role,_=store.remember_stranger(s3,duration_ms=5000,now_ms=3000)
            self.assertTrue(sid.startswith('stranger_'));self.assertEqual(role,'stranger')
            self.assertEqual(store.profiles[sid]['lifecycle'],'established');self.assertEqual(store.profiles[sid]['samples'],3)
            restored=SpeakerProfileStore(Path(tmp)/'voices.json')
            self.assertEqual(restored.match(s2)[0],sid)
            restored.enroll('小天',s2,'owner')
            self.assertEqual(restored.match(s2)[:2],('小天','owner'))
            self.assertIn(sid,restored.profiles['小天']['aliases'])

    def test_short_utterance_does_not_open_a_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=SpeakerProfileStore(Path(tmp)/'voices.json')
            self.assertEqual(store.remember_stranger([0.,1.,0.],duration_ms=1000,now_ms=1)[0],'unknown')
            self.assertEqual(self.strangers(store),[])
            store.remember_stranger([0.,1.,0.],duration_ms=5000,now_ms=1)
            self.assertEqual(len(self.strangers(store)),1)

    def test_gray_zone_blocks_new_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=SpeakerProfileStore(Path(tmp)/'voices.json')
            store.enroll('a',[1.,0.,0.],'owner');store.enroll('b',[0.,1.,0.],'owner')
            mid=[float(np.cos(np.pi/4)),float(np.sin(np.pi/4)),0.]   # ≈.71 to both: no margin, best above gray floor
            self.assertEqual(store.remember_stranger(mid,duration_ms=5000,now_ms=1)[0],'unknown')
            self.assertEqual(self.strangers(store),[])
            self.assertEqual(store.remember_stranger([0.,0.,1.],duration_ms=5000,now_ms=1)[0],'unknown')
            self.assertEqual(len(self.strangers(store)),1)

    def test_provisional_expires(self):
        from voice_detection import speaker_embedding as se
        with tempfile.TemporaryDirectory() as tmp:
            store=SpeakerProfileStore(Path(tmp)/'voices.json')
            store.remember_stranger([1.,0.,0.],duration_ms=5000,now_ms=0)
            old=self.strangers(store)[0]
            store.remember_stranger([0.,1.,0.],duration_ms=5000,now_ms=se.STRANGER_PROVISIONAL_TTL_MS+1)
            self.assertNotIn(old,store.profiles)

    def test_collapsed_pool_freezes_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=SpeakerProfileStore(Path(tmp)/'voices.json')
            for i,sim in enumerate((1.0,.975,.95)):     # pairwise ≈.95–.975: above the .93 collapse bar
                out=store.remember_stranger(self.at(sim),duration_ms=6000,now_ms=i)
            sid=self.strangers(store)[0]
            self.assertEqual(store.profiles[sid]['lifecycle'],'frozen')
            self.assertEqual(out[0],'unknown');self.assertEqual(store.match(self.at(.975))[0],'unknown')

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
