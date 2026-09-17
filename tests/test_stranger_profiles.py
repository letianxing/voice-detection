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


class BodyEvidenceTests(unittest.TestCase):
    """One voice profile must not keep absorbing utterances from two people the camera saw together."""

    @staticmethod
    def voice(sim):
        t=float(np.arccos(sim));return [float(np.cos(t)),float(np.sin(t)),0.]

    def test_two_visible_bodies_sharing_one_voice_contest_the_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=SpeakerProfileStore(Path(tmp)/'v.json')
            s1=[1.,0.,0.];s2=[.85,float(np.sqrt(1-.85**2)),0.];s3=[.85,0.,float(np.sqrt(1-.85**2))]
            store.remember_stranger(s1,duration_ms=5000,now_ms=1,internal_id='p1',visible_ids=['p1','p2'])
            out=store.remember_stranger(s2,duration_ms=5000,now_ms=2,internal_id='p2',visible_ids=['p1','p2'])
            sid=[k for k,v in store.profiles.items() if v.get('role')=='stranger'][0]
            self.assertEqual(store.profiles[sid]['lifecycle'],'contested');self.assertEqual(out[0],'unknown')
            # contested stays unpublished and stops growing
            self.assertEqual(store.remember_stranger(s3,duration_ms=5000,now_ms=3,internal_id='p1',visible_ids=['p1'])[0],'unknown')
            self.assertEqual(store.profiles[sid]['samples'],2)

    def test_sequential_bodies_are_not_a_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=SpeakerProfileStore(Path(tmp)/'v.json')
            s1=[1.,0.,0.];s2=[.85,float(np.sqrt(1-.85**2)),0.];s3=[.85,0.,float(np.sqrt(1-.85**2))]
            store.remember_stranger(s1,duration_ms=5000,now_ms=1,internal_id='track7',visible_ids=['track7'])
            store.remember_stranger(s2,duration_ms=5000,now_ms=2,internal_id='track9',visible_ids=['track9'])   # left and came back: new id
            sid,role,_=store.remember_stranger(s3,duration_ms=5000,now_ms=3,internal_id='track9',visible_ids=['track9'])
            self.assertEqual(role,'stranger');self.assertEqual(store.profiles[sid]['lifecycle'],'established')

    def test_published_profile_is_withdrawn_when_a_second_body_hits_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=SpeakerProfileStore(Path(tmp)/'v.json')
            s1=[1.,0.,0.];s2=[.85,float(np.sqrt(1-.85**2)),0.];s3=[.85,0.,float(np.sqrt(1-.85**2))]
            for i,v in enumerate((s1,s2,s3)):sid,_,_=store.remember_stranger(v,duration_ms=5000,now_ms=i,internal_id='p1',visible_ids=['p1'])
            self.assertEqual(store.profiles[sid]['lifecycle'],'established')
            out=store.remember_stranger(s2,duration_ms=5000,now_ms=9,internal_id='p2',visible_ids=['p1','p2'])
            self.assertEqual(out[0],'unknown');self.assertEqual(store.profiles[sid]['lifecycle'],'contested')
            self.assertEqual(store.match(s2)[0],'unknown')

    def test_unbound_utterances_never_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=SpeakerProfileStore(Path(tmp)/'v.json')
            s1=[1.,0.,0.];s2=[.85,float(np.sqrt(1-.85**2)),0.];s3=[.85,0.,float(np.sqrt(1-.85**2))]
            store.remember_stranger(s1,duration_ms=5000,now_ms=1)
            store.remember_stranger(s2,duration_ms=5000,now_ms=2,internal_id='',visible_ids=['p1','p2'])
            sid,role,_=store.remember_stranger(s3,duration_ms=5000,now_ms=3,internal_id='p1',visible_ids=['p1','p2'])
            self.assertEqual(role,'stranger')


class VisualBindingTests(unittest.TestCase):
    def setUp(self):
        from voice_detection.dashboard import bind_utterance_to_visible_person
        self.bind=bind_utterance_to_visible_person

    def test_bearing_picks_the_one_person_within_window(self):
        people=[{'person_id':'a','azimuth_deg':-20.},{'person_id':'b','azimuth_deg':25.}]
        out=self.bind({'direction_valid':True,'direction_deg':22.},people)
        self.assertEqual((out['internal_id'],out['reason']),('b','bearing'));self.assertEqual(out['visible_ids'],['a','b'])

    def test_two_at_bearing_needs_lips_to_choose(self):
        people=[{'person_id':'a','azimuth_deg':0.,'lip_motion':False,'lip_motion_valid':True},
                {'person_id':'b','azimuth_deg':8.,'lip_motion':True,'lip_motion_valid':True}]
        self.assertEqual(self.bind({'direction_valid':True,'direction_deg':4.},people)['internal_id'],'b')
        people[1]['lip_motion']=False
        out=self.bind({'direction_valid':True,'direction_deg':4.},people)
        self.assertEqual((out['internal_id'],out['reason']),('','several_at_bearing'))

    def test_without_bearing_only_a_lone_lip_moving_person_binds(self):
        lone=[{'person_id':'a','azimuth_deg':0.,'lip_motion':True,'lip_motion_valid':True}]
        self.assertEqual(self.bind({'direction_valid':False},lone)['reason'],'only_person_lips')
        lone[0]['lip_motion_valid']=False
        self.assertEqual(self.bind({'direction_valid':False},lone)['internal_id'],'')
        self.assertEqual(self.bind({'direction_valid':False},[])['reason'],'nobody_visible')

    def test_vision_down_means_no_evidence_not_an_error(self):
        from voice_detection.dashboard import fetch_vision_people
        self.assertEqual(fetch_vision_people('http://127.0.0.1:1/api/state',timeout_s=.05),[])
