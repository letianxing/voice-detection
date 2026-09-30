"""RAM-only audio slices keyed by ADC wall-clock timestamps, no capture owner."""
import base64
from collections import deque
import threading
import numpy as np
from .asr import resample_mono


class AudioWindows:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows = deque(maxlen=200)

    def clear(self):
        with self.lock:
            self.rows.clear()

    def append(self, stamp, samples, rate, track, echo):
        with self.lock:
            if self.rows and stamp<=self.rows[-1][0]:
                self.rows.clear()
            self.rows.append((stamp,np.asarray(samples,dtype=np.float32).copy(),rate,str(track),float(echo)))
            while self.rows and stamp-self.rows[0][0]>3000:
                self.rows.popleft()

    def window(self, end, now):
        if not 0<=now-end<1200:
            return {'available':False,'reason':'stale_or_future_audio_request'}
        start = end-1000
        with self.lock:
            rows = list(self.rows)
        data = np.zeros(16000,np.float32)
        covered = np.zeros(16000,bool)
        tracks = set()
        for stamp,samples,rate,track,echo in rows:
            if stamp>=end or stamp+len(samples)*1000/rate<=start:
                continue
            if not np.isfinite(echo) or echo>=.3 or not track:
                return {'available':False,'reason':'echo_or_unknown_audio_track'}
            tracks.add(track)
            mono = resample_mono(samples,rate,16000)
            offset = round((stamp-start)*16)
            lo,hi = max(0,offset),min(16000,offset+len(mono))
            if hi>lo:
                data[lo:hi] = mono[lo-offset:hi-offset]
                covered[lo:hi] = True
        # Integer millisecond ADC stamps can leave <1ms rounding holes. Fill
        # only those; reject real dropped buffers rather than pad with silence.
        missing = np.flatnonzero(~covered)
        if len(tracks)!=1 or not covered[0] or not covered[-1] or len(missing)>160:
            return {'available':False,'reason':'audio_track_change_or_gap'}
        if len(missing):
            runs = np.split(missing,np.where(np.diff(missing)>1)[0]+1)
            if max(map(len,runs))>16:
                return {'available':False,'reason':'audio_gap'}
            good = np.flatnonzero(covered)
            data[missing] = np.interp(missing,good,data[good])
        if not np.isfinite(data).all():
            return {'available':False,'reason':'invalid_audio'}
        pcm = (np.clip(data,-1,1)*32767).astype('<i2')
        return {'available':True,'track_id':tracks.pop(),'ended_ms':end,'sample_rate':16000,
                'timestamp_source':'adc_wall_clock','interpolated_samples':len(missing),
                'pcm16_base64':base64.b64encode(pcm.tobytes()).decode()}
