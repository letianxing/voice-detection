import unittest
import numpy as np
from voice_detection.doa import estimate_azimuth_gcc
from voice_detection.beamforming import mixdown

class DeadChannelTests(unittest.TestCase):
    def test_single_live_channel_never_invents_60_degree_bearing(self):
        samples=np.zeros((960,8),np.float32);samples[:,0]=np.random.default_rng(1).normal(0,.02,960)
        result=estimate_azimuth_gcc(samples,48000,[[.03*i,0,0] for i in range(8)],range(6))
        self.assertIsNone(result.azimuth_deg);self.assertEqual(result.confidence,0)
        np.testing.assert_allclose(mixdown(samples),samples[:,0])
