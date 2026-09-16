import unittest
import numpy as np
from voice_detection.tone_events import TonePulseDetector

class ToneTests(unittest.TestCase):
    def test_counts_tone_pulses_from_pcm_not_asr(self):
        detector=TonePulseDetector();events=[]
        for stamp in range(0,400,20):
            x=.1*np.sin(2*np.pi*1000*np.arange(320)/16000) if 20<=stamp<100 or 180<=stamp<260 else np.zeros(320)
            result=detector.update(x,16000,stamp)
            if result:events.append(result)
        self.assertEqual(len(events),2)
        self.assertEqual([e['stamp_ms'] for e in events],[20,180])
