import unittest
import base64
import numpy as np
from voice_detection.asd_capture import AudioWindows


class AudioWindowTests(unittest.TestCase):
    def window(self, skip=None, conflict=False, echo=0):
        buffer = AudioWindows()
        for stamp in range(1000,2020,20):
            if stamp == skip:
                continue
            buffer.append(stamp,np.full(320,.25,np.float32),16000,
                          'other' if conflict and stamp==1500 else 't',echo)
        return buffer

    def test_adc_aligned_slice(self):
        buffer = self.window()
        result = buffer.window(2000,2050)
        self.assertTrue(result['available'])
        pcm = np.frombuffer(base64.b64decode(result['pcm16_base64']),dtype='<i2')
        self.assertEqual(len(pcm),16000)
        self.assertTrue(np.all(pcm==8191))
        buffer.clear()
        self.assertFalse(buffer.window(2000,2050)['available'])

    def test_gap_track_switch_echo_and_expiry(self):
        for buffer in (self.window(skip=1500),self.window(conflict=True),self.window(echo=.8)):
            self.assertFalse(buffer.window(2000,2050)['available'])
        self.assertFalse(self.window().window(2000,3500)['available'])
