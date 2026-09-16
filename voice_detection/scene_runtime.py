"""Bounded, asynchronous acoustic/music analysis for the live microphone."""
from collections import deque
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
import numpy as np
from .acoustic_reflex import AcousticReflex, LiveBeatDetector
from .audio_scene import DEFAULT_MODEL, classification_worker
from .doa import estimate_azimuth_gcc, normalize_angle_deg
from .bio_acoustics import BioAcoustics


def empty_music(reason="warming_up"):
    return {"valid": False, "music_state": False, "genre": "unknown", "genre_valid": False,
            "bpm": None, "bpm_valid": False, "reason": reason, "stamp_ms": 0}


class SceneRuntime:
    def __init__(self, profile, classifier=True):
        from scipy.signal import resample_poly
        self._resample_poly = resample_poly
        self.profile = profile
        self.frames = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.state_data = {"running": True, "acoustic": {}, "music": empty_music(), "events": [],
                           "classifier_ready": False, "error": "", "dropped_frames": 0}
        self.bio=BioAcoustics()
        self.stereo_audio=deque(maxlen=200)
        self.turn_predictor=None
        self.last_semantic_alarm=-100000
        self.last_cough=-100000
        self.novelty_stamps=deque(maxlen=32)
        self.last_event_request=0
        self.reflex = AcousticReflex()
        self.beats = LiveBeatDetector()
        self.audio = deque(maxlen=500)
        self.events = deque(maxlen=256)
        self.direction_history = deque(maxlen=200)
        self.run_id = uuid.uuid4().hex[:12]
        self.sequence = 0
        self.last_submit = 0
        self.last_doa = 0
        self.direction = None
        self.direction_confidence = 0.
        self.music_on = False
        self.silent_since = None
        self.requests = self.results = self.process = None
        if classifier and os.environ.get("VOICE_VAP", "1")=="1":
            from .turn_prediction import TurnPredictor
            self.turn_predictor=TurnPredictor()
        if classifier:
            ctx = mp.get_context("spawn")
            self.requests, self.results = ctx.Queue(maxsize=1), ctx.Queue(maxsize=4)
            self.process = ctx.Process(target=classification_worker, args=(self.requests, self.results,
                                       str(os.environ.get("VOICE_AUDIO_MODEL", DEFAULT_MODEL))), daemon=True)
            self.process.start()
        self.thread = threading.Thread(target=self._run, daemon=True, name="voice-scene-fast")
        self.thread.start()

    def submit(self, frame, clean_audio, echo_probability=0., track_id="", vad_active=None, playback_reference=None):
        item = (frame, np.asarray(clean_audio, np.float32).copy(), float(echo_probability), {"track_id":track_id,"vad_active":vad_active,"reference":None if playback_reference is None else np.asarray(playback_reference,np.float32).copy()})
        try:
            self.frames.put_nowait(item)
        except queue.Full:
            with self.lock:
                self.state_data["dropped_frames"] += 1
            try:
                self.frames.get_nowait()
                self.frames.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def snapshot(self):
        with self.lock:
            result = dict(self.state_data)
            result["events"] = list(self.events)
            music = dict(result["music"])
            if music.get("valid") and int(time.time() * 1000) - music.get("stamp_ms", 0) > 8000:
                music = empty_music("stale")
            result["music"] = music
            result["turn_prediction"]=self.turn_predictor.snapshot() if self.turn_predictor else {"valid":False,"ready":False,"reason":"disabled"}
            return result

    def direction_at(self, stamp_ms):
        with self.lock:
            candidates = [item for item in self.direction_history if item.get("direction_valid") and abs(item["stamp_ms"] - stamp_ms) <= 200]
            return dict(min(candidates, key=lambda item: abs(item["stamp_ms"] - stamp_ms))) if candidates else {"direction_valid": False, "direction_deg": None}

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.turn_predictor:self.turn_predictor.close()
        if self.process:
            try: self.requests.put_nowait(None)
            except queue.Full: pass
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
            self.requests.cancel_join_thread()
            self.requests.close()
            self.results.cancel_join_thread()
            self.results.close()
        with self.lock:
            self.state_data.update(running=False, music=empty_music("capture_stopped"))

    def _event(self, event):
        self.sequence += 1
        event["id"] = f"{self.run_id}:{self.sequence}"
        self.events.append(event)

    def _poll_classifier(self):
        if self.results is None:
            return
        while True:
            try: result = self.results.get_nowait()
            except queue.Empty: break
            with self.lock:
                self.state_data["classifier_ready"] = result.get("ready", False)
                self.state_data["error"] = result.get("error", "")
                if result.get("error"):
                    self.state_data["music"] = empty_music("classifier_error")
                elif "music_probability" in result and result.get("stamp_ms", 0) >= self.state_data["music"].get("stamp_ms", 0):
                    self.music_on = result["music_probability"] >= (.35 if self.music_on else .6)
                    result.update(valid=True, music_state=self.music_on, reason="model")
                    if not self.music_on:
                        result.update(genre="unknown", genre_valid=False, bpm=None, bpm_valid=False)
                    self.state_data["music"] = result
                    event_stamp=result.get("event_source_stamp_ms")
                    scores=result.get("event_scores") or {}
                    cough_score=max(float(scores.get("Cough",0)),float(scores.get("Sneeze",0)))
                    now=int(time.time()*1000)
                    if event_stamp and 0<=now-event_stamp<3000 and cough_score>=.8 and now-self.last_cough>=30000:
                        self.last_cough=now
                        self._event({"kind":"cough","stamp_ms":now,"source_stamp_ms":event_stamp,"model_score":cough_score,"reason":"classified_cough_or_sneeze"})
                    hazards={k:v for k,v in scores.items() if k in {"Shatter","Explosion","Gunshot, gunfire","Screaming"}}
                    label,score=max(hazards.items(),key=lambda item:item[1]) if hazards else ("unknown",0.)
                    now=int(time.time()*1000)
                    if (event_stamp and 0<=now-event_stamp<3000 and now-self.last_semantic_alarm>=8000
                            and label in {"Shatter","Explosion","Gunshot, gunfire","Screaming"} and score>=.88 and score>cough_score):
                        direction=self.direction_at(event_stamp)
                        self.last_semantic_alarm=now
                        self._event({"kind":"startle","stamp_ms":now,"source_stamp_ms":event_stamp,"direction_deg":direction.get("direction_deg"),
                                     "direction_valid":bool(direction.get("direction_valid")),"reason":"semantic_alarm_confirmed","event_label":label,
                                     "model_score":score,"classification_delay_ms":now-event_stamp,"event_window_ms":result.get("event_window_ms"),"calibrated":False})
        if self.process is not None and not self.process.is_alive() and not self.stop_event.is_set():
            with self.lock:
                self.state_data["classifier_ready"] = False
                self.state_data["error"] = self.state_data["error"] or "audio classifier process exited"

    def _run(self):
        while not self.stop_event.is_set():
            self._poll_classifier()
            try: frame, audio, echo, context = self.frames.get(timeout=.05)
            except queue.Empty: continue
            try:
                self._process(frame, audio, echo, context)
            except Exception as exc:
                with self.lock:
                    self.state_data["error"] = str(exc)

    def _process(self, frame, audio, echo, context=None):
        from math import gcd
        stamp, rate = frame.stamp_ms, frame.sample_rate_hz
        rms = float(np.sqrt(np.mean(audio * audio)) + 1e-10)
        dbfs = max(-120., 20 * np.log10(rms))
        raw_rms = np.sqrt(np.mean(frame.samples * frame.samples, axis=0))
        can_locate = self.profile.supports_doa and bool(self.profile.doa_channel_indices)
        if dbfs > -60 and can_locate and stamp - self.last_doa >= 40:
            doa = estimate_azimuth_gcc(frame.samples, rate, self.profile.mic_positions_m, self.profile.doa_channel_indices)
            sign = float(os.environ.get("VOICE_DOA_SIGN", "1"))
            offset = float(os.environ.get("VOICE_DOA_OFFSET_DEG", "0"))
            self.direction = normalize_angle_deg(sign * doa.azimuth_deg + offset) if doa.azimuth_deg is not None and doa.confidence >= .35 else None
            self.direction_confidence = doa.confidence
            self.last_doa = stamp
        if not can_locate or stamp - self.last_doa > 200 or dbfs <= -60:
            self.direction, self.direction_confidence = None, 0.
        offset = os.environ.get("VOICE_SPL_CALIBRATION_DB")
        spl_channel = min(len(raw_rms) - 1, max(0, int(os.environ.get("VOICE_SPL_CHANNEL", "0"))))
        raw_dbfs = max(-120., float(20 * np.log10(float(raw_rms[spl_channel]) + 1e-10)))
        raw_indices = self.profile.doa_channel_indices if can_locate else []
        missing_raw = bool(raw_indices and float(raw_rms.max())>1e-4 and any(i>=len(raw_rms) or raw_rms[i]<max(1e-6,float(raw_rms.max())*.01) for i in raw_indices))
        ild=[]
        if can_locate and not missing_raw:
            for index,i in enumerate(raw_indices):
                for j in raw_indices[index+1:]:
                    ild.append({"channels":[i,j],"level_difference_db":round(float(20*np.log10((raw_rms[i]+1e-10)/(raw_rms[j]+1e-10))),2)})
        acoustic = {"pair_ild":ild,"ild_calibrated":False,"direction_method":"gcc_phat_tdoa" if can_locate else "unavailable", "direction_warning": "阵列原始通道缺失或静音，方位不可用" if missing_raw else "", "stamp_ms": stamp, "rms_dbfs": round(float(dbfs), 2),
                    "raw_rms_dbfs": round(raw_dbfs, 2), "spl_channel": spl_channel,
                    "spl_db": round(raw_dbfs + float(offset), 2) if offset else None,
                    "spl_calibrated": bool(offset), "direction_deg": self.direction,
                    "direction_valid": self.direction is not None, "direction_confidence": round(self.direction_confidence, 3),
                    "coordinate_frame": "robot_yaw_deg_front_0_ccw", "channel_rms": raw_rms.tolist(),
                    "echo_probability": echo, "processing_age_ms": max(0, int(time.time() * 1000) - stamp)}
        divisor = gcd(rate, 16000)
        mono = self._resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32) if rate != 16000 else audio
        context=context or {}
        bio=self.bio.update(mono,stamp,context.get("track_id",""),context.get("vad_active"),echo,raw_dbfs)
        acoustic["bio"]=bio
        if bio.get("novelty_onset"):self.novelty_stamps.append(stamp)
        reference=context.get("reference")
        if reference is None:reference=np.zeros_like(mono)
        elif rate!=16000:reference=self._resample_poly(reference,16000//divisor,rate//divisor).astype(np.float32)
        if len(reference)==len(mono):self.stereo_audio.append(np.stack([mono,reference]))
        if self.turn_predictor and sum(x.shape[-1] for x in self.stereo_audio)>=32000:
            self.turn_predictor.submit(np.concatenate(self.stereo_audio,axis=1)[:,-64000:],stamp)
        self.audio.append(mono)
        music = self.snapshot()["music"]
        music_valid = music.get("valid") and music.get("music_state") and echo < .65
        if not hasattr(self,"tone_detector"):
            from .tone_events import TonePulseDetector
            self.tone_detector=TonePulseDetector()
        tone=self.tone_detector.update(mono,16000,stamp) if echo<.4 else None
        with self.lock:
            if tone:self._event(tone)
            self.state_data["acoustic"] = acoustic
            self.direction_history.append(acoustic)
            for event in self.reflex.update(float(dbfs), stamp, self.direction, self.direction_confidence, echo, bool(music_valid and not (bio.get("novelty_onset") and bio.get("energy_z",0)>=8 and bio.get("spectral_flux",0)>=.5))):
                # Non-extreme impulses orient first; semantic classification decides
                # whether a cough or an alarming event occurred, rather than shouting at every cough.
                if event.get("kind")=="startle" and raw_dbfs < -6:
                    event.update(kind="orient",reason="impulse_pending_classification")
                self._event(event)
            beat = self.beats.update(mono, stamp, bool(music_valid and music.get("bpm_valid")), music.get("bpm"), music.get("beat_anchor_ms"))
            if beat:
                self._event(beat)
        if dbfs < -65 or echo >= .8:
            if self.silent_since is None:
                self.silent_since = stamp
            if stamp - self.silent_since >= 500:
                self.audio.clear()
                self.music_on = False
                with self.lock:
                    self.state_data["music"] = {**empty_music("silence_or_self_playback"), "valid": True, "stamp_ms": stamp}
        else:
            self.silent_since = None
        count = sum(chunk.size for chunk in self.audio)
        if self.requests is not None and count >= 16000 * 2 and stamp - self.last_submit >= 1000 and self.state_data["classifier_ready"]:
            try:
                event_stamp=self.novelty_stamps[-1] if self.novelty_stamps and stamp-self.novelty_stamps[-1]<1500 and self.novelty_stamps[-1]!=self.last_event_request else None
                self.requests.put_nowait((np.concatenate(self.audio)[-160000:], stamp,event_stamp))
                if event_stamp:self.last_event_request=event_stamp
                self.last_submit = stamp
            except queue.Full:
                pass
