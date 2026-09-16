"""Persistent low-latency PCM output with timestamped echo reference."""
from __future__ import annotations
import base64
from collections import deque
import threading
import time
import numpy as np
from .aec import PlaybackReferenceBuffer
from .types import AudioFrame


class PlaybackEngine:
    def __init__(self):
        self.reference = PlaybackReferenceBuffer(retention_ms=5000)
        self.lock = threading.RLock()
        self.operation_lock = threading.RLock()
        self.stream = None
        self.rate = 0
        self.device = None
        self.playback_id = ""
        self.turn_id = ""
        self.speaking = False
        self.error = ""
        self.ended_ms = 0
        self.first_audio_ms = None
        self.finish_at = None
        self.samples = np.zeros(0, np.float32)
        self.position = 0
        self.recent = deque(maxlen=40)
        self.fade_start_gain=1.0
        self.fade_total=0
        self.fade_remaining=0
        self.stop_mode="none"

    def _finish_if_due(self):
        if self.finish_at is not None and time.time()*1000 >= self.finish_at:
            self.speaking = False
            self.ended_ms = int(self.finish_at)
            self.finish_at = None
            if self.recent:
                self.recent[-1]["ended_ms"] = self.ended_ms

    def state(self):
        with self.lock:
            self._finish_if_due()
            return {"id": self.playback_id, "turn_id": self.turn_id, "speaking": self.speaking,
                    "error": self.error, "ended_ms": self.ended_ms, "first_audio_ms": self.first_audio_ms,
                    "reference_enabled": True, "output_warm": self.stream is not None,
                    "fade_remaining_ms":self.fade_remaining*1000/max(1,self.rate),"stop_mode":self.stop_mode}

    def warmup(self, rate=16000):
        import sounddevice as sd
        with self.operation_lock:
            device = int(sd.query_devices(kind="output").get("index", sd.default.device[1]))
            if self.stream is not None and self.rate == rate and self.device == device:
                return
            self.close()
            self.rate, self.device = rate, device
            self.stream = sd.OutputStream(device=device, samplerate=rate, channels=1, dtype="float32",
                                          blocksize=rate//100, latency="low", callback=self._render)
            try:
                self.stream.start()
            except Exception:
                self.stream.close()
                self.stream = None
                raise

    def _render(self,outdata,frames,timing,status):
        outdata.fill(0)
        rate=self.rate
        stamp=int((time.time()+timing.outputBufferDacTime-timing.currentTime)*1000)
        with self.lock:
            self._finish_if_due()
            if status:self.error=str(status)
            if not self.speaking or self.position>=len(self.samples):return
            count=min(frames,len(self.samples)-self.position)
            outdata[:count,0]=self.samples[self.position:self.position+count]
            if self.position==0:self.first_audio_ms=stamp
            self.position+=count
            played=count
            if self.fade_remaining:
                n=min(count,self.fade_remaining)
                offset=self.fade_total-self.fade_remaining
                phase=(np.arange(n,dtype=np.float32)+offset)/max(1,self.fade_total-1) if self.fade_total>1 else np.ones(n,np.float32)
                outdata[:n,0]*=self.fade_start_gain*.5*(1+np.cos(np.pi*phase))
                outdata[n:,0]=0
                self.fade_remaining-=n
                if self.fade_remaining==0:
                    self.position=len(self.samples);played=n
            if self.position>=len(self.samples):self.finish_at=stamp+played*1000/rate
        # AEC must see the actual faded waveform, not the original PCM.
        self.reference.append(AudioFrame(outdata.copy(),rate,stamp))

    def start(self, payload):
        rate = int(payload["sample_rate_hz"])
        if rate not in (16000,22050,24000,44100,48000):
            raise ValueError("unsupported playback sample rate")
        pcm = base64.b64decode(payload["pcm_s16le"], validate=True)
        if not pcm or len(pcm)>rate*2*180 or len(pcm)%2:
            raise ValueError("playback must contain 0–180 seconds of mono PCM16")
        samples = np.frombuffer(pcm,dtype="<i2").astype(np.float32)/32768
        with self.operation_lock:
            self.stop()
            self.warmup(rate)
            with self.lock:
                self.playback_id, self.turn_id = str(payload["id"]), str(payload.get("turn_id") or "")
                self.samples, self.position = samples, 0
                self.first_audio_ms, self.finish_at = None, None
                self.fade_total,self.fade_remaining,self.stop_mode=0,0,"none"
                self.error, self.speaking = "", True
                self.recent.append({"id": self.playback_id, "turn_id": self.turn_id, "text": str(payload.get("text") or ""), "started_ms": int(time.time()*1000), "ended_ms": None})
        return self.state()

    def stop(self, key=None, fade_ms=0):
        with self.operation_lock:
            with self.lock:
                if key and key != self.playback_id:
                    return self.state()
                self._finish_if_due()
                was_speaking = self.speaking
                fade_ms=max(0,min(200,float(fade_ms)))
                if was_speaking and fade_ms and self.stream is not None and self.position<len(self.samples):
                    samples=max(1,min(len(self.samples)-self.position,int(self.rate*fade_ms/1000)))
                    if self.fade_remaining:
                        # Repeated stop/finally requests never restart or lengthen a fade.
                        if samples<self.fade_remaining:
                            progress=(self.fade_total-self.fade_remaining)/max(1,self.fade_total-1)
                            self.fade_start_gain*=.5*(1+np.cos(np.pi*progress))
                            self.fade_total=self.fade_remaining=samples
                    else:
                        self.fade_start_gain=1.0
                        self.fade_total=self.fade_remaining=samples
                    self.stop_mode="fade"
                    return self.state()
                self.fade_total,self.fade_remaining=0,0
                self.stop_mode="immediate"
                self.speaking, self.finish_at = False, None
                self.samples, self.position = np.zeros(0,np.float32), 0
                self.ended_ms = int(time.time()*1000)
                if self.recent and was_speaking:
                    self.recent[-1]["ended_ms"] = self.ended_ms
            if self.stream is not None and was_speaking:
                # Abort buffered output immediately, then keep the device warm for the next phrase.
                self.stream.abort()
                self.stream.start()
        return self.state()

    def close(self):
        with self.operation_lock:
            with self.lock:
                stream, self.stream = self.stream, None
                self.speaking, self.finish_at = False, None
                self.samples = np.zeros(0,np.float32)
                self.fade_total,self.fade_remaining=0,0
            if stream is not None:
                stream.abort()
                stream.close()

    def echo_match(self, text, started_ms, ended_ms, speaker_similarity=0.):
        from .echo_guard import match_playback
        with self.lock:
            records = [dict(item) for item in self.recent]
        return match_playback(text, started_ms, ended_ms, records, speaker_similarity)

    def aligned_reference(self, frame):
        """Estimate acoustic delay over 0–240ms using decimated correlation."""
        from .types import AudioFrame
        rate = frame.sample_rate_hz
        lag_samples = int(rate * .24)
        window = AudioFrame(np.zeros((len(frame.samples) + lag_samples, 1), np.float32),
                            rate, frame.stamp_ms - 240)
        history = self.reference.read_for(window)
        mic = np.mean(frame.samples, axis=1)
        stride = max(1, rate // 4000)
        x, h = mic[::stride], history[::stride]
        x = x - x.mean()
        if np.dot(h, h) < 1e-8 or np.dot(x, x) < 1e-8:
            return None
        correlation = np.correlate(h, x, mode="valid")
        energies = np.convolve(h * h, np.ones(len(x)), mode="valid")
        scores = abs(correlation) / np.sqrt(np.maximum(energies * np.dot(x, x), 1e-12))
        offset = int(np.argmax(scores)) * stride
        return history[offset:offset + len(mic)]
