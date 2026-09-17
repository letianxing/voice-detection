from __future__ import annotations

import argparse
import signal
from collections import deque
from .playback import PlaybackEngine
from .scene_runtime import SceneRuntime, empty_music
from .ros_publisher import BrainRosPublisher
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .frontend import CocktailFrontend, now_ms
from .asr import ExternalCommandAsrAdapter, NullAsrAdapter, SherpaZipformerAsrAdapter, VoskAsrAdapter, write_wav
from .io import list_sounddevice_devices
from .profiles import get_profile, load_profiles
from .ros4hri import acoustic_track_to_ros4hri_json
from .types import AudioFrame
from .streaming_asr import AsrWorker, UtteranceSegmenter
from .utterance_finalizer import UtteranceFinalizer
from .target_speaker import create_target_speaker_extractor
from .speaker_embedding import SpeakerEmbedder, SpeakerProfileStore


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>voice-detection</title>
  <style>
    body { margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f7fa; color: #172033; }
    header { padding: 18px 24px; background: #ffffff; border-bottom: 1px solid #d8dee8; }
    main { max-width: 1080px; margin: 0 auto; padding: 20px; display: grid; gap: 16px; }
    section { background: #ffffff; border: 1px solid #d8dee8; border-radius: 8px; padding: 16px; }
    label { display: block; font-weight: 650; margin-bottom: 6px; }
    select, input, button { font: inherit; padding: 8px 10px; border: 1px solid #b9c3d3; border-radius: 6px; background: #fff; }
    button { cursor: pointer; background: #143d79; color: white; border-color: #143d79; }
    button.secondary { background: #fff; color: #172033; border-color: #b9c3d3; }
    .row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; align-items: end; }
    .metrics { display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 10px; }
    .metric { border: 1px solid #e1e6ee; border-radius: 6px; padding: 10px; min-height: 58px; }
    .metric strong { display: block; font-size: 22px; }
    pre { white-space: pre-wrap; word-break: break-word; background: #111827; color: #dbeafe; padding: 12px; border-radius: 6px; min-height: 140px; }
    @media (max-width: 780px) { .row, .metrics { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header><h1>voice-detection</h1></header>
  <main>
    <section>
      <div class="row">
        <div><label for="profile">Profile</label><select id="profile"></select></div>
        <div><label for="device">Device</label><select id="device"></select></div>
        <div><label for="tse">Cocktail mode</label><select id="tse"><option value="baseline">Baseline DOA/BF</option><option value="speakerbeam">SpeakerBeam TSE</option><option value="auto">Auto overlap switch</option><option value="multi">多人分段分离（需注册参考）</option></select></div>
        <div><label for="seconds">Seconds</label><input id="seconds" value="0" inputmode="numeric"></div>
        <div>
          <button id="start">Start</button>
          <button id="stop" class="secondary">Stop</button>
        </div>
      </div>
    </section>
    <section class="metrics">
      <div class="metric">Track<strong id="track">-</strong></div>
      <div class="metric">VAD<strong id="vad">-</strong></div>
      <div class="metric">音乐/反射/ROS<strong id="sceneSummary">--</strong></div>
      <div class="metric">Azimuth<strong id="az">-</strong></div>
      <div class="metric">Clarity<strong id="clarity">-</strong></div>
      <div class="metric">Overlap<strong id="overlap">-</strong></div>
      <div class="metric">Speaker<strong id="speaker">-</strong></div>
      <div class="metric">Similarity<strong id="similarity">-</strong></div>
      <div class="metric">ASR partial<strong id="partial">-</strong></div>
      <div class="metric">TSE backend<strong id="tseBackend">-</strong></div>
      <div class="metric">TSE latency<strong id="tseLatency">-</strong></div>
    </section>
    <section><pre id="log"></pre></section>
  </main>
  <script>
    const profile = document.getElementById("profile");
    const device = document.getElementById("device");
    const log = document.getElementById("log");
    async function loadOptions() {
      const profiles = await (await fetch("/api/profiles")).json();
      profile.innerHTML = profiles.map(p => `<option value="${p.id}">${p.id} - ${p.label}</option>`).join("");
      const devices = await (await fetch("/api/devices")).json();
      device.innerHTML = devices.map(d => d.error
        ? `<option value="default">${d.error}</option>`
        : `<option value="${d.index}">${d.index} - ${d.name} (${d.max_input_channels}ch)</option>`).join("");
      if (!device.innerHTML) device.innerHTML = `<option value="default">default</option>`;
    }
    async function poll() {
      const state = await (await fetch("/api/state")).json();
      const track = state.last_track || {};
      document.getElementById("track").textContent = track.track_id || "-";
      document.getElementById("vad").textContent = track.voice_activity === undefined ? "-" : String(track.voice_activity);
      document.getElementById("az").textContent = track.azimuth_deg === null || track.azimuth_deg === undefined ? "-" : `${track.azimuth_deg.toFixed(1)} deg`;
      document.getElementById("clarity").textContent = track.clarity === undefined ? "-" : track.clarity.toFixed(2);
      document.getElementById("overlap").textContent = track.overlap_probability === undefined ? "-" : track.overlap_probability.toFixed(2);
      const speaker = state.last_speaker || {};
      document.getElementById("speaker").textContent = speaker.speaker_id || track.speaker_label || "-";
      document.getElementById("similarity").textContent = speaker.similarity === undefined ? "-" : Number(speaker.similarity).toFixed(2);
      const scene = state.audio_scene || {}; const music = scene.music || {};
      document.getElementById("sceneSummary").textContent = `${music.valid ? (music.music_state ? "音乐" : "非音乐") : "待分析"} / ${music.genre || "unknown"} / BPM ${music.bpm_valid ? music.bpm : "--"} / 反射 ${(scene.events || []).filter(e=>e.kind!=="music_beat").length} / ROS ${(state.ros || {}).connected ? "已连接" : "未连接"} ${scene.error || ""}`;
      const streaming = state.last_streaming_transcript || {};
      document.getElementById("partial").textContent = streaming.text || "-";
      document.getElementById("tseBackend").textContent = state.last_tse?.backend || state.tse_backend || "-";
      document.getElementById("tseLatency").textContent = state.last_tse ? `${Number(state.last_tse.latency_ms || 0).toFixed(1)} ms` : "-";
      log.textContent = JSON.stringify(state, null, 2);
    }
    document.getElementById("start").onclick = async () => {
      const params = new URLSearchParams({profile: profile.value, device: device.value, tse: document.getElementById("tse").value, seconds: document.getElementById("seconds").value});
      await fetch(`/api/start?${params.toString()}`, {method: "POST"});
      poll();
    };
    document.getElementById("stop").onclick = async () => { await fetch("/api/stop", {method: "POST"}); poll(); };
    loadOptions().then(poll);
    setInterval(poll, 350);
  </script>
</body>
</html>
"""


VISION_STATE_URL = os.environ.get("VOICE_VISION_STATE_URL", "http://127.0.0.1:8080/api/state")
VISUAL_BIND_DEG = float(os.environ.get("VOICE_VISUAL_BIND_DEG", "15"))          # pending calibration
VISUAL_AZIMUTH_SIGN = float(os.environ.get("VOICE_VISUAL_AZIMUTH_SIGN", "1"))  # -1 if camera and array bearings are mirrored
VISUAL_AZIMUTH_OFFSET_DEG = float(os.environ.get("VOICE_VISUAL_AZIMUTH_OFFSET_DEG", "0"))


def fetch_vision_people(url=None, timeout_s=0.2):
    """People visible to vision-detection right now, or [] when it is not running.

    Called once per finished utterance from the finalizer thread, never from the audio
    callback. Vision being down is normal (voice runs alone in tests); it only means
    no body evidence for this utterance.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(url or VISION_STATE_URL, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except Exception:
        return []
    people = (payload.get("state") or payload).get("people") or []
    return [p for p in people if isinstance(p, dict) and p.get("person_id")]


def bind_utterance_to_visible_person(direction, people):
    """Which visible person this utterance came from, for the stranger-profile guard only.

    Returns {"internal_id": person_id or "", "visible_ids": [...], "reason": ...}. This
    does not decide who is being spoken to or overwrite anyone's face: it records which
    body the sound lined up with so that one voice profile cannot keep absorbing
    utterances from two people the camera saw together.

    Rules, in order: sound bearing within VISUAL_BIND_DEG of exactly one person; if
    several, the one whose lips were moving (if exactly one); no bearing, exactly one
    person in view with valid moving lips; otherwise unbound.
    """
    visible = [str(p.get("person_id")) for p in people]
    result = {"internal_id": "", "visible_ids": visible, "reason": "unbound"}
    if not people:
        result["reason"] = "nobody_visible"
        return result
    moving = [p for p in people if p.get("lip_motion_valid") and p.get("lip_motion")]
    if direction.get("direction_valid") and direction.get("direction_deg") is not None:
        bearing = VISUAL_AZIMUTH_SIGN * float(direction["direction_deg"]) + VISUAL_AZIMUTH_OFFSET_DEG
        near = [p for p in people if p.get("has_azimuth", True) and p.get("azimuth_deg") is not None
                and abs(float(p["azimuth_deg"]) - bearing) <= VISUAL_BIND_DEG]
        if len(near) == 1:
            return dict(result, internal_id=str(near[0]["person_id"]), reason="bearing")
        if len(near) > 1:
            near_moving = [p for p in near if p in moving]
            if len(near_moving) == 1:
                return dict(result, internal_id=str(near_moving[0]["person_id"]), reason="bearing_and_lips")
            result["reason"] = "several_at_bearing"
            return result
        result["reason"] = "nobody_at_bearing"
        return result
    if len(people) == 1 and moving:
        return dict(result, internal_id=str(people[0]["person_id"]), reason="only_person_lips")
    result["reason"] = "no_bearing"
    return result


class LiveMonitor:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.error = ""
        self.last_track: dict[str, object] | None = None
        self.last_vad: dict[str, object] | None = None
        self.last_speaker: dict[str, object] | None = None
        self.speaker_embedder = None
        self.speaker_profiles = None
        self.last_event: dict[str, object] | None = None
        self._stream = None
        self._stop_timer: threading.Timer | None = None
        self._segmenter: UtteranceSegmenter | None = None
        self._asr_worker: AsrWorker | None = None
        self._asr_adapter = None
        self._last_partial_text = ""
        self._finalizer = None
        self._capture_generation = 0
        self.last_transcript: dict[str, object] | None = None
        self.last_streaming_transcript: dict[str, object] | None = None
        self._streaming_asr = None
        self._streaming_adapter = None
        self._streaming_started_ms = 0
        self.asr_error = ""
        self.asr_backend = "none"
        self._sample_rate_hz = 48000
        self.robot_speaking = False
        self.playback = PlaybackEngine()
        self.transcripts = deque(maxlen=256)
        self.utterance_context = {}
        self._utterance_echo = []
        self._profile_id = None
        self.input_device_info: dict[str, object] | None = None
        self.tse_mode = "baseline"
        self.target_speaker_extractor = create_target_speaker_extractor(self.tse_mode)
        self.last_tse: dict[str, object] | None = None
        self._last_utterance_audio = None
        self._last_utterance_rate = 0
        self.scene = None
        self.ros = None
        if os.environ.get("VOICE_ROSBRIDGE_URL"):
            self.ros = BrainRosPublisher(self.state, os.environ["VOICE_ROSBRIDGE_URL"])

    def start(self, profile_id: str, device: str, seconds: float, tse_mode: str = "baseline") -> None:
        self.stop()
        self.asr_error = ""
        try:
            import numpy as np
            import sounddevice as sd
        except Exception as exc:
            with self.lock:
                self.error = f"sounddevice is required for live capture: {exc}"
            return

        self.multi_transcriber=None
        self.tse_mode = str(tse_mode or "baseline")
        self.target_speaker_extractor = create_target_speaker_extractor(self.tse_mode)
        profile = get_profile(profile_id)
        self._profile_id = profile_id
        self._sample_rate_hz = profile.sample_rate_hz
        frontend = CocktailFrontend(profile, vad_calibration_frames=15)
        model_path = os.environ.get(
            "VOICE_SPEAKER_MODEL",
            str(Path(__file__).resolve().parents[1] / "weights" / "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"),
        )
        self.speaker_embedder = SpeakerEmbedder(model_path)
        self.speaker_profiles = SpeakerProfileStore(
            os.environ.get(
                "VOICE_SPEAKER_PROFILE_STORE",
                str(Path(__file__).resolve().parents[1] / "config" / "speaker_profiles.json"),
            )
        )
        self._segmenter = UtteranceSegmenter(
            pre_roll_ms=int(os.environ.get("VOICE_PRE_ROLL_MS", "180")),
            endpoint_silence_ms=int(os.environ.get("VOICE_ENDPOINT_SILENCE_MS", "480")),
            min_speech_ms=int(os.environ.get("VOICE_MIN_SPEECH_MS", "80")),
        )
        command = os.environ.get("VOICE_ASR_COMMAND", "").strip()
        vosk_model = os.environ.get("VOICE_ASR_VOSK_MODEL", "").strip()
        sherpa_model = os.environ.get("VOICE_ASR_SHERPA_MODEL", "").strip()
        if sherpa_model:
            adapter = SherpaZipformerAsrAdapter(sherpa_model, os.environ.get("VOICE_ASR_LANGUAGE", "zh-CN"))
            self.asr_backend = "sherpa-zipformer-zh"
        elif command:
            adapter = ExternalCommandAsrAdapter(
                command,
                os.environ.get("VOICE_ASR_LANGUAGE", "zh"),
                int(os.environ.get("VOICE_ASR_SAMPLE_RATE", "16000")),
            )
            self.asr_backend = "whisper-cli"
        elif vosk_model:
            adapter = VoskAsrAdapter(vosk_model, os.environ.get("VOICE_ASR_LANGUAGE", "zh-CN"))
            self.asr_backend = "vosk"
        else:
            adapter = NullAsrAdapter()
            self.asr_backend = "none"
        # Whisper is used for the final utterance when configured.  Keep a
        # small Vosk recognizer alongside it so the dashboard can still show
        # low-latency partial text without downgrading final accuracy.
        self._streaming_adapter = None
        if sherpa_model:
            self._streaming_adapter = adapter
        elif command and vosk_model:
            try:
                self._streaming_adapter = VoskAsrAdapter(vosk_model, os.environ.get("VOICE_ASR_LANGUAGE", "zh-CN"))
            except Exception:
                self._streaming_adapter = None
        elif hasattr(adapter, "create_stream"):
            self._streaming_adapter = adapter
        self._asr_adapter = adapter
        self._asr_worker = AsrWorker(adapter, self._on_transcript, self._on_asr_error, self._preprocess_tse_audio)
        self._finalizer = UtteranceFinalizer(self._finalize_utterance, self._on_asr_error)
        self._streaming_asr = None
        if isinstance(adapter, SherpaZipformerAsrAdapter):
            adapter.warmup()
        blocksize = int(profile.sample_rate_hz * profile.preferred_block_ms / 1000.0)
        device_arg = None if device == "default" else coerce_device(device)
        try:
            import sounddevice as sd
            info = sd.query_devices(device_arg, "input")
            if int(info["max_input_channels"]) < profile.input_channels:
                raise RuntimeError(
                    f"selected input has {int(info['max_input_channels'])} channels; "
                    f"profile requires {profile.input_channels}"
                )
            self.input_device_info = {
                "index": int(info.get("index", device_arg if isinstance(device_arg, int) else -1)),
                "name": str(info["name"]),
                "max_input_channels": int(info["max_input_channels"]),
                "default_samplerate": float(info["default_samplerate"]),
            }
        except Exception as exc:
            with self.lock:
                self.error = f"input device preflight failed: {exc}"
            return

        try:
            self.playback.warmup()
        except Exception as exc:
            self.playback.error = str(exc)
        if os.environ.get("VOICE_AUDIO_SCENE", "1") == "1":
            self.scene = SceneRuntime(profile)

        def callback(indata, frames, callback_time, status):
            del frames
            event = {"stamp_ms": int((time.time() + callback_time.inputBufferAdcTime - callback_time.currentTime) * 1000), "status": str(status) if status else ""}
            frame = AudioFrame(
                samples=np.asarray(indata, dtype=np.float32).copy(),
                sample_rate_hz=profile.sample_rate_hz,
                stamp_ms=event["stamp_ms"],
            )
            reference = self.playback.aligned_reference(frame)
            output = frontend.process(frame, reference)
            echo = output.tracks[0].self_echo_probability if output.tracks else 0.0
            if self.scene is not None:
                self.scene.submit(frame, output.target_audio, output.tracks[0].raw_echo_probability if output.tracks else echo, track_id=output.target_track_id, vad_active=output.vad_active, playback_reference=self.playback.reference.read_for(frame))
            human_voice = bool(output.vad_active and echo < 0.75)
            if output.vad_active:
                self._utterance_echo.append(echo)
                self._utterance_echo = self._utterance_echo[-1500:]
            speaker_embedding = self.last_speaker.get("embedding") if isinstance(self.last_speaker, dict) else None
            tse_result = self.target_speaker_extractor.process(output.target_audio, frame.sample_rate_hz, speaker_embedding)
            target_audio = tse_result.audio
            track = output.tracks[0] if output.tracks else None
            if track is not None and self.tse_mode in {"speakerbeam", "auto"}:
                track = replace(track, target_speaker_probability=float(tse_result.target_probability),
                                tse_enabled=bool(tse_result.enabled), tse_healthy=bool(tse_result.healthy),
                                tse_latency_ms=float(tse_result.latency_ms),
                                target_speech_rejected=bool(tse_result.target_probability < 0.35))
            utterance = self._segmenter.process(
                target_audio,
                frame.sample_rate_hz,
                frame.stamp_ms,
                bool(track and human_voice),
                track.track_id if track else "",
            )
            feed_audio = target_audio
            if track and human_voice and self._streaming_asr is None and self._streaming_adapter is not None:
                self._streaming_started_ms = frame.stamp_ms
                self._last_partial_text = ""
                self._streaming_asr = self._streaming_adapter.create_stream(track.track_id, frame.sample_rate_hz)
                feed_audio = self._segmenter.stream_prefix()
                with self.lock:
                    self.last_streaming_transcript = {"track_id": track.track_id, "text": "", "is_final": False,
                                                      "started_ms": frame.stamp_ms, "emitted_ms": None, "first_result_latency_ms": None}
            if self._streaming_asr is not None:
                try:
                    partial = self._streaming_asr.accept(feed_audio, frame.sample_rate_hz, frame.stamp_ms)
                except Exception as exc:
                    partial = ""
                    self._on_asr_error(f"streaming_asr: {exc}")
                if partial and partial != self._last_partial_text:
                    self._last_partial_text = partial
                    partial_echo = self.playback.echo_match(partial, self._streaming_started_ms, frame.stamp_ms)
                    with self.lock:
                        emitted_ms = now_ms()
                        previous_latency = (self.last_streaming_transcript or {}).get("first_result_latency_ms")
                        self.last_streaming_transcript = {
                            "track_id": self._streaming_asr.track_id, "text": partial, "is_final": False,
                            "source": "robot_echo" if partial_echo else "microphone", "echo_match": partial_echo,
                            "started_ms": self._streaming_started_ms, "emitted_ms": emitted_ms,
                            "first_result_latency_ms": previous_latency if previous_latency is not None else max(0, emitted_ms-self._streaming_started_ms)}
            if utterance is None and not self._segmenter.active and self._streaming_asr is not None:
                discarded, self._streaming_asr = self._streaming_asr, None
                self._streaming_started_ms = 0
                UtteranceFinalizer.close_stream(discarded)
            if utterance is not None:
                stream, self._streaming_asr = self._streaming_asr, None
                self._streaming_started_ms = 0
                context = {
                    "_capture_generation": self._capture_generation,
                    "first_result_latency_ms": (self.last_streaming_transcript or {}).get("first_result_latency_ms"),
                    "direction": self.scene.direction_at(utterance.ended_ms) if self.scene else {"direction_valid": False, "direction_deg": None},
                    "overlap_probability": float(track.overlap_probability if track else 0),
                    "self_echo_probability": float(sum(self._utterance_echo) / max(1, len(self._utterance_echo))),
                }
                self._utterance_echo = []
                self._finalizer.submit(utterance, stream, track, context)
            with self.lock:
                self.last_event = event
                self.last_track = track.to_json_dict() if track else None
                self.last_vad = {
                    "active": output.vad_active,
                    "probability": output.vad_probability,
                    "rms_dbfs": output.rms_dbfs,
                    "noise_floor_dbfs": output.noise_floor_dbfs,
                    "snr_db": output.snr_db,
                    "agc_gain_db": output.agc_gain_db,
                    "stamp_ms": frame.stamp_ms,
                }
                self.last_tse = {
                    "backend": tse_result.backend,
                    "enabled": tse_result.enabled,
                    "healthy": tse_result.healthy,
                    "target_probability": tse_result.target_probability,
                    "latency_ms": tse_result.latency_ms,
                }

        try:
            self._stream = sd.InputStream(
                device=device_arg,
                channels=profile.input_channels,
                samplerate=profile.sample_rate_hz,
                blocksize=blocksize,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()
            with self.lock:
                self.running = True
                self.error = ""
                self.last_event = {"stamp_ms": now_ms(), "started": True, "profile": profile_id, "device": device}
                self.last_event["input_device_info"] = self.input_device_info
            if seconds > 0:
                self._stop_timer = threading.Timer(seconds, self.stop)
                self._stop_timer.daemon = True
                self._stop_timer.start()
        except Exception as exc:
            with self.lock:
                self.running = False
                self.error = str(exc)
            self.stop()

    def _finalize_utterance(self, utterance, stream, track, context):
        context = dict(context)
        generation = context.pop("_capture_generation", self._capture_generation)
        asr_worker = self._asr_worker
        if generation != self._capture_generation:
            return
        self._last_utterance_key = (utterance.started_ms, utterance.ended_ms)
        self._last_utterance_audio = utterance.samples.copy()
        self._last_utterance_rate = utterance.sample_rate_hz
        context["visual"] = bind_utterance_to_visible_person(
            context.get("direction") or {}, fetch_vision_people())
        streamed_final = None
        if stream is not None:
            try:
                streamed_final = stream.finish(utterance.ended_ms)
            except Exception as exc:
                if generation == self._capture_generation:
                    self._on_asr_error(f"streaming_asr_finish: {exc}")
        if generation != self._capture_generation or self._finalizer is None or self._finalizer.stop_event.is_set():
            return
        if self.tse_mode=="multi":
            from .multi_speaker import MultiSpeakerTranscriber
            if self.multi_transcriber is None:
                self.multi_transcriber=MultiSpeakerTranscriber(self.speaker_profiles,self.speaker_embedder,self._asr_adapter,Path(__file__).resolve().parents[1]/"weights/real-tse/pretrained/spk_emb_causal_100")
            try:
                result=self.multi_transcriber.transcribe(utterance.samples,utterance.sample_rate_hz,utterance.started_ms,utterance.ended_ms,self._finalizer.stop_event)
                self.last_tse={k:v for k,v in result.items() if k!="turns"}
                self.last_tse["mixed_asr"]=streamed_final.text if streamed_final else ""
                if generation!=self._capture_generation:return
                if result.get("ready") and result.get("turns"):
                    from .types import SpeechTranscript
                    for separated in result["turns"]:
                        identity=separated["person_id"]
                        profile=self.speaker_profiles.profiles.get(identity,{})
                        key=(utterance.track_id,utterance.started_ms)
                        self.utterance_context[key]={"separation":{"verified":True,"speaker_id":identity,"mode":"completed_segment","latency_ms":result.get("latency_ms"),"word_timing_available":False},
                            "speaker":{"speaker_id":identity,"speaker_role":profile.get("role","known"),"similarity":separated["identity_score"],"track_id":utterance.track_id,"stamp_ms":utterance.ended_ms}}
                        self._on_transcript(SpeechTranscript(utterance.track_id,separated["text"],True,language="zh-CN",started_ms=utterance.started_ms,ended_ms=utterance.ended_ms,emitted_ms=now_ms()))
                    return
            except Exception as exc:
                self.last_tse={"ready":False,"error":str(exc),"mode":"completed_segment"}
            # Keep fallback words, but never infer identity from the inseparable mix.
            if streamed_final is not None:
                self.utterance_context[(utterance.track_id,utterance.started_ms)]={"separation":{"verified":False,"mode":"mixed_fallback"},"overlap_probability":1.0}
                self._on_transcript(streamed_final)
            return
        embedding_start = time.perf_counter()
        embedding = self.speaker_embedder.embed(utterance.samples, utterance.sample_rate_hz) if self.speaker_embedder else None
        context["speaker_embedding_ms"] = round((time.perf_counter()-embedding_start)*1000,2)
        context["asr_decode_ms"] = getattr(stream, "decode_ms", None)
        if generation != self._capture_generation:
            return
        speaker = {"speaker_id": "unknown", "speaker_role": "unknown", "similarity": 0,
                   "track_id": utterance.track_id, "stamp_ms": utterance.ended_ms}
        if embedding:
            speaker_id, role, similarity = self.speaker_profiles.match(embedding)
            speaker.update(speaker_id=speaker_id, speaker_role=role, similarity=similarity, embedding=embedding)
            ranked=self.speaker_profiles.rank(embedding)
            speaker["pool_size"]=ranked[0]["pool_size"] if ranked else 0
            speaker["match_margin"]=ranked[0]["score"]-ranked[1]["score"] if len(ranked)>1 else None
        with self.lock:
            self.last_speaker = speaker
            self.utterance_context[(utterance.track_id, utterance.started_ms)] = dict(context, speaker=dict(speaker))
            while len(self.utterance_context) > 256:
                self.utterance_context.pop(next(iter(self.utterance_context)))
        if streamed_final is not None:
            streamed_final = replace(streamed_final, track_id=utterance.track_id, started_ms=utterance.started_ms,
                                     ended_ms=utterance.ended_ms, emitted_ms=now_ms(), clarity=float(track.clarity if track else .7))
            with self.lock:
                if self._streaming_started_ms <= utterance.ended_ms:
                    self.last_streaming_transcript = {**asdict(streamed_final), "final_latency_ms": max(0, now_ms() - utterance.ended_ms), "first_result_latency_ms": context.get("first_result_latency_ms")}
            if self.asr_backend in {"sherpa-zipformer-zh", "vosk"} and self.tse_mode == "baseline":
                self._on_transcript(streamed_final)
                return
        if asr_worker is not None and generation == self._capture_generation:
            asr_worker.submit(utterance)

    def stop(self) -> None:
        self._capture_generation += 1
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._segmenter is not None and self._asr_worker is not None:
            utterance = self._segmenter.flush(self._sample_rate_hz, now_ms())
            if utterance is not None:
                self._asr_worker.submit(utterance)
        if self._finalizer is not None:
            self._finalizer.close()
            self._finalizer = None
        UtteranceFinalizer.close_stream(self._streaming_asr)
        if self._asr_worker is not None:
            self._asr_worker.close()
            self._asr_worker = None
        if self._asr_adapter is not None and hasattr(self._asr_adapter, "close"):
            self._asr_adapter.close()
        self._asr_adapter = None
        if self.scene is not None:
            self.scene.close()
            self.scene = None
        self._segmenter = None
        self._streaming_asr = None
        self._streaming_adapter = None
        with self.lock:
            self.running = False

    def state(self, compact=False) -> dict[str, object]:
        with self.lock:
            return {
                "running": self.running,
                "profile": self._profile_id,
                "audio_scene": self.scene.snapshot() if self.scene else {"running": False, "music": empty_music("capture_stopped")},
                "ros": self.ros.status() if self.ros else {"enabled": False},
                "error": self.error,
                "last_track": self.last_track,
                "last_vad": self.last_vad,
                "last_speaker": self.last_speaker,
                "last_event": self.last_event,
                "last_transcript": self.last_transcript,
                "active_utterance": self._segmenter.snapshot() if self._segmenter else None,
                "transcripts": [{key: value for key, value in item.items() if key != "speaker_vector"} for item in self.transcripts] if compact else list(self.transcripts),
                "playback": self.playback.state(),
                "last_streaming_transcript": self.last_streaming_transcript,
                "asr_enabled": bool(
                    os.environ.get("VOICE_ASR_SHERPA_MODEL", "").strip()
                    or os.environ.get("VOICE_ASR_COMMAND", "").strip()
                    or os.environ.get("VOICE_ASR_VOSK_MODEL", "").strip()
                ),
                "asr_error": self.asr_error,
                "asr_backend": self.asr_backend,
                "asr_model_ready": bool(self._asr_adapter.engine.ready.is_set()) if isinstance(self._asr_adapter, SherpaZipformerAsrAdapter) and self._asr_adapter.engine else self.asr_backend not in {"none", "sherpa-zipformer-zh"},
                "asr_decode_ms": self._asr_adapter.engine.last_decode_ms if isinstance(self._asr_adapter, SherpaZipformerAsrAdapter) and self._asr_adapter.engine else None,
                "speaker_model_ready": bool(self.speaker_embedder and self.speaker_embedder.ready),
                "speaker_model_error": self.speaker_embedder.error if self.speaker_embedder else "",
                "finalizer_dropped": self._finalizer.dropped if self._finalizer else 0,
                "tse_backend": getattr(self.target_speaker_extractor, "backend", "baseline"),
                "tse_mode": self.tse_mode,
                "last_tse": self.last_tse,
                "robot_speaking": self.playback.speaking or self.robot_speaking,
                "input_device_info": self.input_device_info,
            }

    def set_robot_speaking(self, speaking: bool) -> None:
        with self.lock:
            self.robot_speaking = bool(speaking)

    def _on_transcript(self, transcript) -> None:
        with self.lock:
            data = asdict(transcript)
            context = self.utterance_context.pop((transcript.track_id, transcript.started_ms), {})
            speaker = context.pop("speaker", {})
            data["speaker_vector"] = speaker.pop("embedding", [])
            data.update(context)
            echo_match = self.playback.echo_match(data.get("text", ""), int(data.get("started_ms") or 0), int(data.get("ended_ms") or now_ms()), float(speaker.get("similarity") or 0))
            data["source"] = "robot_echo" if echo_match else "microphone"
            data["echo_match"] = echo_match
            if echo_match:
                speaker = dict(speaker, speaker_id="robot", speaker_role="robot")
            if (not echo_match and data.get("speaker_vector") and self.speaker_profiles
                    and float(data.get("self_echo_probability") or 0)<.4 and float(data.get("overlap_probability") or 0)<.25
                    and transcript.ended_ms-transcript.started_ms>=1500 and str(data.get("text") or "").strip()):
                visual = data.get("visual") or {}
                identity, role, similarity=self.speaker_profiles.remember_stranger(
                    data["speaker_vector"], duration_ms=transcript.ended_ms-transcript.started_ms, now_ms=transcript.ended_ms,
                    internal_id=visual.get("internal_id", ""), visible_ids=visual.get("visible_ids") or [])
                speaker.update(speaker_id=identity,speaker_role=role,similarity=similarity)
                self.last_speaker=dict(speaker,embedding=data["speaker_vector"],finalized_identity=True)
            from .echo_guard import normalized
            profile = self.speaker_profiles.profiles.get(speaker.get("speaker_id"), {}) if self.speaker_profiles else {}
            if speaker:
                speaker["aliases"] = profile.get("aliases", [])
            if self.last_speaker and self.last_speaker.get("finalized_identity") and self.last_speaker.get("stamp_ms")==speaker.get("stamp_ms"):
                self.last_speaker["identity_embedding"] = profile.get("embedding", [])
                self.last_speaker["aliases"] = profile.get("aliases", [])
            registered_text = normalized(profile.get("enrollment_text", ""))
            data["repeats_enrollment"] = bool(len(registered_text) >= 6 and registered_text == normalized(data.get("text", "")) and not echo_match)
            data["speaker"] = speaker
            data["utterance_id"] = f"{transcript.track_id}:{transcript.started_ms}:{transcript.ended_ms}"
            if data.get("separation",{}).get("verified"):
                data["utterance_id"] += ":"+str(data["separation"]["speaker_id"])
            self.last_transcript = data
            self.transcripts.append(data)
            self.asr_error = ""

    def _on_asr_error(self, error: str) -> None:
        with self.lock:
            self.asr_error = error

    def _preprocess_tse_audio(self, audio, sample_rate_hz):
        if self.tse_mode not in {"speakerbeam", "auto"}:
            return audio
        result = self.target_speaker_extractor.process_utterance(audio, sample_rate_hz)
        with self.lock:
            self.last_tse = {
                "backend": result.backend, "enabled": result.enabled, "healthy": result.healthy,
                "target_probability": result.target_probability, "latency_ms": round(result.latency_ms, 2),
            }
        return result.audio

    def save_tse_reference(self, speaker_id=None) -> bool:
        if self._last_utterance_audio is None or not self._last_utterance_rate:
            return False
        path = Path(__file__).resolve().parents[1] / "config" / "speaker_references" / "target.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(path, self._last_utterance_audio, self._last_utterance_rate)
        if speaker_id:
            import hashlib
            named=path.parent/(hashlib.sha256(speaker_id.encode()).hexdigest()+".wav")
            write_wav(named,self._last_utterance_audio,self._last_utterance_rate)
            self.speaker_profiles.profiles[speaker_id]["reference_path"]=str(named)
            self.speaker_profiles.save()
        self.target_speaker_extractor = create_target_speaker_extractor(self.tse_mode)
        return True


def make_handler(monitor: LiveMonitor):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_text(INDEX_HTML, "text/html; charset=utf-8")
            elif parsed.path == "/api/profiles":
                profiles = [profile.__dict__ for profile in load_profiles().values()]
                self._send_json(profiles)
            elif parsed.path == "/api/devices":
                self._send_json(list_sounddevice_devices())
            elif parsed.path == "/api/playback":
                self._send_json(monitor.playback.state())
            elif parsed.path == "/api/state":
                self._send_json(monitor.state(compact=parse_qs(parsed.query).get("compact") == ["1"]))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path in {"/api/playback/start", "/api/playback/stop"}:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length > 18_000_000:
                        raise ValueError("playback request too large")
                    payload = json.loads(self.rfile.read(length) or "{}")
                    result = monitor.playback.start(payload) if parsed.path.endswith("/start") else monitor.playback.stop(payload.get("id"), payload.get("fade_ms",20))
                    self._send_json(result)
                except Exception as exc:
                    self._send_json({"error": str(exc)})
            elif parsed.path == "/api/start":
                monitor.start(
                    profile_id=query.get("profile", ["mac_builtin"])[0],
                    device=query.get("device", ["default"])[0],
                    tse_mode=query.get("tse", ["baseline"])[0],
                    seconds=float(query.get("seconds", ["0"])[0] or 0),
                )
                self._send_json(monitor.state())
            elif parsed.path == "/api/stop":
                monitor.stop()
                self._send_json(monitor.state())
            elif parsed.path == "/api/robot-speaking":
                monitor.set_robot_speaking(query.get("speaking", ["0"])[0] in {"1", "true", "True"})
                self._send_json(monitor.state())
            elif parsed.path == "/api/voice-pool":
                payload=json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))) or "{}")
                with monitor.lock:
                    sample=monitor.last_transcript or {}
                    valid=(sample.get("utterance_id")==payload.get("utterance_id") and 0<=now_ms()-int(sample.get("ended_ms") or 0)<3000 and sample.get("source")!="robot_echo")
                    accepted=False
                    if valid and sample.get("speaker_vector"):
                        accepted=monitor.speaker_profiles.add_verified_sample(str(payload.get("person_id")),sample["speaker_vector"],str(payload.get("face_person_id")),float(payload.get("face_confidence") or 0),
                            {"overlap":float(sample.get("overlap_probability") or 0),"echo":float(sample.get("self_echo_probability") or 0),"duration_ms":int(sample.get("ended_ms") or 0)-int(sample.get("started_ms") or 0)})
                    if accepted and not monitor.speaker_profiles.profiles[str(payload.get("person_id"))].get("reference_path") and getattr(monitor,"_last_utterance_key",None)==(sample.get("started_ms"),sample.get("ended_ms")):
                        monitor.save_tse_reference(str(payload["person_id"]))
                    self._send_json({"accepted":accepted})
            elif parsed.path == "/api/enroll-speaker":
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8") or "{}")
                speaker_id = str(payload.get("speaker_id") or "").strip()
                if not speaker_id or not monitor.last_speaker or not monitor.last_speaker.get("embedding"):
                    self._send_json({"success": False, "message": "speak first, then provide speaker_id"})
                    return
                with monitor.lock:
                    sample = monitor.last_transcript or {}
                    requested = payload.get("utterance_id")
                    if requested and (sample.get("utterance_id") != requested or sample.get("source") == "robot_echo"
                                      or not sample.get("speaker_vector") or sample.get("ended_ms") != monitor.last_speaker.get("stamp_ms")
                                      or now_ms()-sample.get("ended_ms",0)>15000):
                        self._send_json({"success": False, "message": "声音样本已变化或无效，请重新说一段话"})
                        return
                    embedding = sample["speaker_vector"] if requested else monitor.last_speaker["embedding"]
                    if requested:
                        if monitor._last_utterance_key != (sample.get("started_ms"),sample.get("ended_ms")):
                            self._send_json({"success":False,"message":"声音样本已更新，请再完整说一次"});return
                        try:
                            embedding=monitor.speaker_embedder.enrollment_embedding(monitor._last_utterance_audio,monitor._last_utterance_rate)
                        except ValueError as exc:
                            self._send_json({"success":False,"message":str(exc)});return
                    if payload.get("dry_run"):
                        self._send_json({"success":True,"speaker_id":speaker_id});return
                    monitor.speaker_profiles.enroll(speaker_id, embedding, str(payload.get("speaker_role") or "known"), sample.get("text", "") if requested else "")
                    monitor.last_speaker["speaker_id"] = speaker_id
                    monitor.last_speaker["speaker_role"] = str(payload.get("speaker_role") or "known")
                    monitor.last_speaker["similarity"] = 1.0
                    monitor.last_speaker["identity_embedding"] = list(embedding)
                    monitor.last_speaker["aliases"] = monitor.speaker_profiles.profiles[speaker_id].get("aliases", [])
                    monitor.last_speaker["finalized_identity"] = True
                reference_saved = monitor.save_tse_reference(speaker_id)
                self._send_json({"success": True, "speaker_id": speaker_id, "tse_reference_saved": reference_saved})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, fmt, *args):  # noqa: A003
            return

        def _send_json(self, data):
            self._send_text(json.dumps(data, ensure_ascii=False), "application/json; charset=utf-8")

        def _send_text(self, text: str, content_type: str):
            body = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(host: str, port: int) -> None:
    monitor = LiveMonitor()
    server = ThreadingHTTPServer((host, port), make_handler(monitor))
    try:
        print(f"voice dashboard: http://{host}:{port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        if monitor.ros:
            monitor.ros.close()
        monitor.playback.close()
        server.server_close()


def coerce_device(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
