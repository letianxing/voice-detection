import unittest
import numpy as np
from voice_detection.bio_acoustics import BioAcoustics
from voice_detection.acoustic_reflex import AcousticReflex

class BioTests(unittest.TestCase):
    def test_pitch_is_in_hz_and_silence_is_not_voiced(self):
        b=BioAcoustics()
        for stamp in range(1000,1800,20):
            t=(np.arange(320)+(stamp-1000)*16)/16000
            r=b.update(.1*np.sin(2*np.pi*200*t),stamp,'speaker',True)
        self.assertAlmostEqual(r['f0_hz'],200,delta=5)
        self.assertGreater(r['voicing_confidence'],.8)
        for stamp in range(1800,2000,20):r=b.update(np.zeros(320),stamp,'speaker',False)
        self.assertIsNone(r['f0_hz']);self.assertFalse(r['breath_detection_supported'])
    def test_roughness_measures_modulation_not_carrier_band(self):
        values=[]
        for modulated in [False,True]:
            b=BioAcoustics()
            for stamp in range(1000,1800,20):
                t=(np.arange(320)+(stamp-1000)*16)/16000
                x=.08*np.sin(2*np.pi*3000*t)*(1+.8*np.sin(2*np.pi*70*t) if modulated else 1)
                r=b.update(x,stamp)
            values.append(r['roughness_30_150hz'])
        self.assertLess(values[0],.1);self.assertGreater(values[1],.7)
    def test_moderate_transient_attracts_without_forcing_startle(self):
        b=BioAcoustics();reflex=AcousticReflex()
        for stamp in range(0,3000,20):b.update(np.zeros(320),stamp);reflex.update(-80,stamp,None,0)
        x=np.random.default_rng(4).normal(0,.025,320)
        r=b.update(x,3000)
        self.assertTrue(r['novelty_onset'])
        db=20*np.log10(np.sqrt(np.mean(x*x)))
        self.assertEqual(reflex.update(db,3000,None,0),[])
