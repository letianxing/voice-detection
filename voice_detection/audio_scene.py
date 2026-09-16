"""AudioSet model inference and tempo estimation. No heuristic genre labels."""
from __future__ import annotations
import os
from pathlib import Path
import time
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "weights" / "ast-audioset"
def summarize_scores(scores):
    return {"music_probability": float(scores.get("Music", 0)),
            "speech_probability": max(float(scores.get(k, 0)) for k in ("Speech", "Conversation", "Narration, monologue")),
            "genre": "unknown", "genre_valid": False, "genre_confidence": 0., "genre_candidates": [],
            "event_scores": {k:float(scores.get(k,0)) for k in ("Shatter","Smash, crash","Explosion","Gunshot, gunfire","Screaming","Crying, sobbing","Cough","Sneeze")},
            "top_events": sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]}


def estimate_tempo(audio, rate=16000):
    import librosa
    if len(audio) < rate * 4 or np.sqrt(np.mean(np.square(audio))) < 1e-5:
        return {"bpm": None, "bpm_valid": False, "bpm_confidence": 0., "beat_times_sec": []}
    onset = librosa.onset.onset_strength(y=audio, sr=rate, hop_length=256, n_fft=1024)
    bpm, beats = librosa.beat.beat_track(onset_envelope=onset, sr=rate, hop_length=256, trim=False)
    bpm = float(np.asarray(bpm).reshape(-1)[0])
    intervals = np.diff(beats) * 256 / rate
    if len(intervals) < 3 or not 40 <= bpm <= 240:
        return {"bpm": None, "bpm_valid": False, "bpm_confidence": 0., "beat_times_sec": []}
    centered = onset - np.mean(onset)
    lag = int(round(60 * rate / (bpm * 256)))
    coherence = float(np.dot(centered[:-lag], centered[lag:]) / (np.dot(centered, centered) + 1e-9)) if 0 < lag < len(centered) else 0
    regularity = max(0., 1. - float(np.std(intervals) / (np.mean(intervals) + 1e-9)))
    confidence = max(0., min(1., coherence * regularity))
    valid = confidence >= .2
    return {"bpm": round(bpm, 2) if valid else None, "bpm_valid": valid,
            "bpm_confidence": round(confidence, 3), "beat_times_sec": (beats * 256 / rate).tolist() if valid else []}


class AudioSetClassifier:
    def __init__(self, model_dir=None):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        path = str(model_dir or os.environ.get("VOICE_AUDIO_MODEL", DEFAULT_MODEL))
        torch.set_num_threads(int(os.environ.get("VOICE_AUDIO_THREADS", "2")))
        self.processor = ASTFeatureExtractor.from_pretrained(path, local_files_only=True)
        self.model = ASTForAudioClassification.from_pretrained(path, local_files_only=True, use_safetensors=True).eval()
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        genre_path = str(os.environ.get("VOICE_GENRE_MODEL", DEFAULT_MODEL.parent / "music-genre"))
        self.genre_processor = AutoFeatureExtractor.from_pretrained(genre_path, local_files_only=True)
        self.genre_model = AutoModelForAudioClassification.from_pretrained(genre_path, local_files_only=True, use_safetensors=True).eval()

    def classify(self, audio, rate=16000):
        import torch
        if rate != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            divisor = gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        model_started = time.perf_counter()
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            probabilities = self.model(**inputs).logits.sigmoid()[0].cpu().numpy()
        result = summarize_scores({self.model.config.id2label[i]: float(value) for i, value in enumerate(probabilities)})
        result["model_timings_ms"] = {"AST": round((time.perf_counter()-model_started)*1000,2)}
        result.update(genre="unknown", genre_valid=False, genre_confidence=0., genre_candidates=[])
        if result["music_probability"] >= .35:
            genre_started = time.perf_counter()
            features = self.genre_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.inference_mode():
                scores = self.genre_model(**features).logits.softmax(dim=-1)[0].cpu().numpy()
            result["model_timings_ms"]["Genre"] = round((time.perf_counter()-genre_started)*1000,2)
            ranked = sorted([{"genre": self.genre_model.config.id2label[i], "score": float(score)} for i, score in enumerate(scores)],
                            key=lambda item: item["score"], reverse=True)
            valid = ranked[0]["score"] >= float(os.environ.get("VOICE_GENRE_MIN_CONFIDENCE", ".90")) and ranked[0]["score"] - ranked[1]["score"] >= .15
            result.update(genre=ranked[0]["genre"] if valid else "unknown", genre_valid=valid,
                          genre_confidence=ranked[0]["score"], genre_candidates=ranked[:3])
        return result


def classification_worker(requests, results, model_dir):
    try:
        model = AudioSetClassifier(model_dir)
        # Warm feature extraction and JIT tempo kernels outside real-time capture.
        model.classify(np.zeros(16000, np.float32))
        results.put({"ready": True})
    except Exception as exc:
        results.put({"ready": False, "error": str(exc)})
        return
    while True:
        job = requests.get()
        if job is None:
            return
        audio, stamp_ms = job[:2]
        event_stamp=job[2] if len(job)>2 else None
        start = time.monotonic()
        try:
            result = model.classify(audio)
            if event_stamp is not None:
                import torch
                focus=audio[-32000:]
                event_started=time.perf_counter()
                inputs=model.processor(focus,sampling_rate=16000,return_tensors="pt")
                with torch.inference_mode():scores=model.model(**inputs).logits.sigmoid()[0].cpu().numpy()
                result["event_scores"]=summarize_scores({model.model.config.id2label[i]:float(v) for i,v in enumerate(scores)})["event_scores"]
                result["event_window_ms"]=len(focus)*1000/16000
                result["event_source_stamp_ms"]=event_stamp
                result["model_timings_ms"]["AST event"]=round((time.perf_counter()-event_started)*1000,2)
            tempo_started = time.perf_counter()
            result.update(estimate_tempo(audio))
            result["model_timings_ms"]["BPM"] = round((time.perf_counter()-tempo_started)*1000,2)
            result["window_ms"] = round(len(audio)*1000/16000)
            if result.get("bpm_valid") and result.get("beat_times_sec"):
                result["beat_anchor_ms"] = stamp_ms - len(audio) * 1000 / 16000 + result["beat_times_sec"][-1] * 1000
            result.update(stamp_ms=stamp_ms, inference_ms=round((time.monotonic() - start) * 1000), ready=True)
            results.put(result)
        except Exception as exc:
            results.put({"ready": True, "stamp_ms": stamp_ms, "error": str(exc)})
