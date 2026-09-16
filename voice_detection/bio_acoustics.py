"""Measured salience/prosody features, not diagnoses or calibrated probabilities.

Inspired by auditory saliency and temporal roughness research. Thresholds are
engineering defaults; the 2-4 kHz energy fraction alone never triggers an alarm.
"""
from collections import deque
import numpy as np


class BioAcoustics:
    def __init__(self):
        self.levels=deque(maxlen=250);self.envelopes=deque(maxlen=400);self.audio=deque(maxlen=4)
        self.spectrum=None;self.previous_db=None;self.previous_stamp=None
        self.last_pitch_ms=0;self.f0=None;self.voicing=0.;self.pitch_history=deque(maxlen=25)
        self.source=None;self.silent_since=None;self.last_novelty=-100000

    def update(self,audio,stamp,source='',vad_active=None,echo=0.,raw_dbfs=None):
        x=np.asarray(audio,dtype=np.float32).reshape(-1)
        if len(x)<32:return {'valid':False,'stamp_ms':stamp,'reason':'short_frame'}
        if self.previous_stamp is not None and stamp-self.previous_stamp>200:
            self.spectrum=None;self.previous_db=None;self.audio.clear();self.envelopes.clear();self.pitch_history.clear()
        rms=float(np.sqrt(np.mean(x*x))+1e-10);db=max(-120.,20*np.log10(rms))
        baseline=float(np.median(self.levels)) if self.levels else db
        mad=float(np.median(abs(np.asarray(self.levels)-baseline))) if self.levels else 1.
        z=(db-baseline)/max(3.,1.4826*mad)
        dt=max(1,stamp-(self.previous_stamp or stamp-20))
        rise=0. if self.previous_db is None else max(0.,db-self.previous_db)
        spectrum=abs(np.fft.rfft(x*np.hanning(len(x)),n=512))
        spectrum/=max(float(spectrum.sum()),1e-9)
        flux=0. if self.spectrum is None else float(np.maximum(spectrum-self.spectrum,0).sum())
        frequencies=np.fft.rfftfreq(512,1/16000)
        band=float(spectrum[(frequencies>=2000)&(frequencies<=4000)].sum())
        n=len(x)//16
        if n:self.envelopes.extend(np.sqrt(np.mean(x[:n*16].reshape(n,16)**2,axis=1)).tolist())
        roughness=None
        if len(self.envelopes)>=200:
            env=np.asarray(self.envelopes);env-=env.mean()
            power=abs(np.fft.rfft(env*np.hanning(len(env))))**2
            modulation=np.fft.rfftfreq(len(env),.001)
            total=float(power[(modulation>=1)&(modulation<=200)].sum())
            roughness=float(power[(modulation>=30)&(modulation<=150)].sum()/total) if total>1e-10 else 0.
        self.audio.append(x)
        if source and source!=self.source:
            self.pitch_history.clear();self.source=source
        if stamp-self.last_pitch_ms>=40:
            signal=np.concatenate(list(self.audio))[-1024:];signal=signal-signal.mean()
            self.f0=None;self.voicing=0.
            if len(signal)>=640 and rms>.003:
                size=len(signal);fft=np.fft.rfft(signal,n=2*size)
                corr=np.fft.irfft(abs(fft)**2)[:size]
                energy=np.concatenate(([0.],np.cumsum(signal*signal)))
                lags=np.arange(1,min(229,size-2)+1)
                diff=np.maximum(0,energy[size-lags]+energy[size]-energy[lags]-2*corr[lags])
                cmnd=diff*lags/np.maximum(np.cumsum(diff),1e-10)
                candidates=[i for i in range(31,len(cmnd)-1) if cmnd[i]<.2 and cmnd[i]<=cmnd[i-1] and cmnd[i]<=cmnd[i+1]]
                if candidates:
                    i=candidates[0];self.f0=float(16000/lags[i]);self.voicing=float(1-cmnd[i])
                    self.pitch_history.append((stamp,self.f0))
            self.last_pitch_ms=stamp
        while self.pitch_history and stamp-self.pitch_history[0][0]>600:self.pitch_history.popleft()
        slope=None
        if len(self.pitch_history)>=4 and self.pitch_history[-1][0]-self.pitch_history[0][0]>=120:
            ts=np.array([t for t,f in self.pitch_history],float)/1000
            pitches=np.array([f for t,f in self.pitch_history]);slope=float(np.polyfit(ts-ts[0],12*np.log2(pitches),1)[0])
        active=(db>-48) if vad_active is None else bool(vad_active)
        if active:self.silent_since=None
        elif self.silent_since is None:self.silent_since=stamp
        residual_fraction=10**((db-raw_dbfs)/20) if raw_dbfs is not None else 1.
        external=echo<.65 or (residual_fraction>.35 and flux>.5)
        novel=(len(self.levels)>=25 and z>=4 and rise>=8 and flux>=.25 and db>=-38 and external and stamp-self.last_novelty>=1200)
        if novel:self.last_novelty=stamp
        score=min(1.,max(0.,.45*min(1,z/8)+.35*min(1,flux/.5)+.2*min(1,rise/20)))
        self.levels.append(float(max(-80,db)));self.previous_db=db;self.previous_stamp=stamp;self.spectrum=spectrum
        return {'valid':True,'stamp_ms':stamp,'source_track_id':source,'baseline_dbfs':round(baseline,2),
                'energy_z':round(float(z),3),'rise_db':round(rise,2),'rise_db_per_second':round(rise*1000/dt,2),
                'spectral_flux':round(flux,4),'energy_2_4khz_fraction':round(band,4),'roughness_30_150hz':roughness,
                'novelty_onset':bool(novel),'salience_score':round(score,3),'external_residual':bool(external),
                'f0_hz':self.f0,'voicing_confidence':self.voicing,'f0_slope_semitones_per_second':slope,
                'pause_ms':0 if self.silent_since is None else stamp-self.silent_since,
                'breath_detection_supported':False,'emotion_inference_supported':False,'calibrated':False}
