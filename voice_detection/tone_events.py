"""Timestamped narrow-band tone pulses from actual PCM; not speech or startle."""
import numpy as np

class TonePulseDetector:
    def __init__(self):
        self.started=None;self.last=None;self.frequency=None
    def update(self,audio,rate,stamp):
        x=np.asarray(audio,dtype=np.float32).reshape(-1)
        if len(x)<16:return None
        power=abs(np.fft.rfft(x*np.hanning(len(x))))**2
        index=int(np.argmax(power));frequency=index*rate/len(x)
        concentration=float(power[max(0,index-1):index+2].sum()/max(float(power.sum()),1e-12))
        tonal=np.sqrt(np.mean(x*x))>.01 and 500<=frequency<=4000 and concentration>.85
        if tonal:
            if self.started is None or (self.last and stamp-self.last>60) or abs(frequency-(self.frequency or frequency))>150:
                self.started=stamp
            self.last=stamp;self.frequency=frequency
            return None
        start=self.started;last=self.last;freq=self.frequency
        self.started=None;self.last=None
        if start is not None and last is not None and 40<=stamp-start<=250:
            return {'kind':'tone_pulse','stamp_ms':start,'ended_ms':stamp,'frequency_hz':freq,'evidence':'narrow_band_pcm'}
        return None
