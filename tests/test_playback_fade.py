import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from voice_detection.playback import PlaybackEngine
from voice_detection.types import AudioFrame

class FadeTests(unittest.TestCase):
    def engine(self,size=1600):
        e=PlaybackEngine();e.stream=Mock();e.rate=16000;e.playback_id='p';e.speaking=True;e.samples=np.ones(size,np.float32)
        return e
    def test_fade_is_smooth_and_reference_matches_actual_output(self):
        e=self.engine();e.stop('p',20);chunks=[]
        for i in range(2):
            out=np.zeros((160,1),np.float32)
            with patch('voice_detection.playback.time.time',return_value=1000+i*.01):e._render(out,160,SimpleNamespace(outputBufferDacTime=0,currentTime=0),None)
            chunks.append(out.copy())
        audio=np.concatenate(chunks)[:,0]
        self.assertAlmostEqual(float(audio[0]),1);self.assertAlmostEqual(float(audio[-1]),0,places=6)
        self.assertTrue(np.all(np.diff(audio)<=1e-7));self.assertLess(float(abs(np.diff(audio)).max()),.01)
        reference=e.reference.read_for(AudioFrame(np.zeros((320,1),np.float32),16000,1000000))
        np.testing.assert_allclose(reference,audio,atol=1e-6)
        with patch('voice_detection.playback.time.time',return_value=1000.03):self.assertFalse(e.state()['speaking'])
        e.stream.abort.assert_not_called()
    def test_duplicate_stop_does_not_restart_fade_and_old_id_is_ignored(self):
        e=self.engine();e.stop('old',20);self.assertEqual(e.fade_remaining,0)
        e.stop('p',20)
        with patch('voice_detection.playback.time.time',return_value=1000):e._render(np.zeros((160,1),np.float32),160,SimpleNamespace(outputBufferDacTime=0,currentTime=0),None)
        remaining=e.fade_remaining;e.stop('p',20);self.assertEqual(e.fade_remaining,remaining)
    def test_fade_shortens_to_remaining_clip(self):
        e=self.engine(32);e.stop('p',40);out=np.zeros((160,1),np.float32)
        with patch('voice_detection.playback.time.time',return_value=1000):e._render(out,160,SimpleNamespace(outputBufferDacTime=0,currentTime=0),None)
        self.assertAlmostEqual(float(out[31,0]),0,places=6);self.assertTrue(np.all(out[32:]==0))
