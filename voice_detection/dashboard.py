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
from .asr import (ExternalCommandAsrAdapter, NullAsrAdapter, SherpaSenseVoiceAsrAdapter,
                  SherpaZipformerAsrAdapter, VoskAsrAdapter, offline_pass_is_corroborated,
                  resample_mono, write_wav)
from .io import list_sounddevice_devices
from .profiles import get_profile, load_profiles
from .ros4hri import acoustic_track_to_ros4hri_json
from .types import AudioFrame
from .streaming_asr import AsrWorker, UtteranceSegmenter
from .utterance_finalizer import UtteranceFinalizer
from .target_speaker import create_target_speaker_extractor
from .speaker_embedding import SpeakerEmbedder, SpeakerProfileStore
from .speaker_tracks import (AuditoryAttentionTarget, MixedAudioRing, SpeakerReferenceResolver,
                             SpeakerTracker)


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
        # A capture request outlives one failed attempt: the array drops off the
        # USB bus mid-session and has to be able to come back by itself.
        self.desired: dict[str, object] | None = None
        self._supervisor: threading.Thread | None = None
        self.retry_count = 0
        self.reopen_count = 0
        self._dead_channel_strikes = 0
        self._last_reopen_ms = 0
        self.capture_warning = ""
        self.last_transcript: dict[str, object] | None = None
        self.last_streaming_transcript: dict[str, object] | None = None
        # Declared here so the state endpoint works before capture has started.
        self.audio_ring = None
        self.speaker_tracker = None
        self.reference_resolver = None
        # Who attention has told us to listen to. The ear does not choose.
        self.attention_target = AuditoryAttentionTarget()
        self.speaker_error = ""
        self._speaker_context = None
        self._speech_frames = deque(maxlen=400)
        self._speaker_worker = None
        self._speaker_stop = threading.Event()
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
        """Ask for capture and keep asking while the device is not there.

        A USB array that drops off the bus used to leave this permanently dead:
        the request failed once, the panel showed a PortAudio number, and the
        service sat there with no microphone until somebody restarted it by
        hand. The request is now a standing one — plugging the array back in
        brings capture back without touching the service.
        """
        self.stop()
        with self.lock:
            self.desired = {"profile": str(profile_id), "device": str(device),
                            "seconds": float(seconds or 0), "tse": str(tse_mode or "baseline")}
        self._start_once(profile_id, device, seconds, tse_mode)
        if not self.running:
            self._ensure_supervisor()

    def _ensure_supervisor(self) -> None:
        if self._supervisor is not None and self._supervisor.is_alive():
            return
        self._supervisor = threading.Thread(target=self._supervise, daemon=True)
        self._supervisor.start()

    def _supervise(self, interval_s: float = 3.0) -> None:
        """Keep a standing capture request true: retry it, and repair it."""
        while True:
            time.sleep(interval_s)
            with self.lock:
                wanted = dict(self.desired) if self.desired else None
                running = self.running
            if wanted is None:
                return
            if running:
                self._repair_dead_channels(wanted)
                continue
            refresh_audio_devices()
            self.retry_count += 1
            self._start_once(wanted["profile"], wanted["device"], wanted["seconds"], wanted["tse"])

    # An array stream opened at the wrong moment — right after the device
    # re-enumerated on USB — delivers channel 0 and digital silence on the rest.
    # Nothing recovers by itself: direction, beamforming and array echo
    # cancellation stay off for the whole session, and the only sign is one line
    # of warning text. Reopening the stream fixed it every time it happened, so
    # the service now does that instead of the person at the keyboard.
    DEAD_CHANNEL_STRIKES = 2
    REOPEN_INTERVAL_MS = 60000

    def _repair_dead_channels(self, wanted) -> None:
        if self.scene is None or not getattr(self.scene.profile, "supports_raw_multichannel", False):
            return
        levels = ((self.scene.snapshot().get("acoustic") or {}).get("channel_rms")) or []
        if len(levels) < 2:
            return
        loudest = max(levels)
        if loudest <= 1e-4:
            self._dead_channel_strikes = 0
            return
        alive = sum(1 for level in levels if level > loudest * .01)
        if alive > 1:
            self._dead_channel_strikes = 0
            return
        self._dead_channel_strikes += 1
        if self._dead_channel_strikes < self.DEAD_CHANNEL_STRIKES:
            return
        if self._last_reopen_ms and now_ms() - self._last_reopen_ms < self.REOPEN_INTERVAL_MS:
            return
        self._dead_channel_strikes = 0
        self._last_reopen_ms = now_ms()
        self.reopen_count += 1
        self._teardown()
        refresh_audio_devices()
        self._start_once(wanted["profile"], wanted["device"], wanted["seconds"], wanted["tse"])
        with self.lock:
            self.capture_warning = (f"阵列只有 1 个通道有信号，已第 {self.reopen_count} 次重开采集流恢复多通道；"
                                    f"设备刚插上时打开的流会出现这种状态。")

    def _start_once(self, profile_id: str, device: str, seconds: float, tse_mode: str = "baseline") -> None:
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
        sense_voice = os.environ.get("VOICE_ASR_SENSEVOICE_MODEL", "").strip()
        # SenseVoice is offline, so it can only be the final pass. When it is
        # configured the streaming model keeps its job — partials for turn-taking
        # and barge-in — and only the finished sentence, which is what reaches
        # memory and the model, gets the more accurate recogniser.
        if sense_voice:
            adapter = SherpaSenseVoiceAsrAdapter(sense_voice, os.environ.get("VOICE_ASR_LANGUAGE", "zh-CN"))
            self.asr_backend = "sensevoice-small" + ("+sherpa-partial" if sherpa_model else "")
        elif sherpa_model:
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
        if sense_voice and sherpa_model:
            self._streaming_adapter = SherpaZipformerAsrAdapter(
                sherpa_model, os.environ.get("VOICE_ASR_LANGUAGE", "zh-CN"))
        elif sherpa_model:
            self._streaming_adapter = adapter
        elif vosk_model and (command or sense_voice):
            try:
                self._streaming_adapter = VoskAsrAdapter(vosk_model, os.environ.get("VOICE_ASR_LANGUAGE", "zh-CN"))
            except Exception:
                self._streaming_adapter = None
        elif hasattr(adapter, "create_stream"):
            self._streaming_adapter = adapter
        self._asr_adapter = adapter
        # The ear keeps the last half minute of what it heard: focused hearing
        # has to be able to start mid-sentence, and speaker tracking reads short
        # windows out of it rather than holding its own copy.
        self.audio_ring = MixedAudioRing(16000, float(os.environ.get("VOICE_RING_SECONDS", "30")))
        self.speaker_tracker = SpeakerTracker(self.speaker_embedder, self.speaker_profiles) \
            if self.speaker_embedder is not None else None
        self.reference_resolver = SpeakerReferenceResolver(
            self.speaker_profiles, Path(__file__).resolve().parents[1]/"config"/"speaker_references"/"session",
            self.audio_ring)
        self._speaker_worker = None
        self._asr_worker = AsrWorker(adapter, self._on_transcript, self._on_asr_error, self._preprocess_tse_audio)
        self._finalizer = UtteranceFinalizer(self._finalize_utterance, self._on_asr_error)
        self._streaming_asr = None
        for warmable in (adapter, self._streaming_adapter):
            if isinstance(warmable, (SherpaZipformerAsrAdapter, SherpaSenseVoiceAsrAdapter)):
                warmable.warmup()
        self._speaker_context = None
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
            # Capturing an eight-microphone array through the single-channel
            # laptop profile is silently legal and throws away everything the
            # array is for: no beamforming, no direction, no array AEC. It
            # happened in the field and nothing said so.
            spare = int(info["max_input_channels"]) - int(profile.input_channels)
            self.capture_warning = (
                f"所选设备 {info['name']} 有 {int(info['max_input_channels'])} 个输入通道，"
                f"当前配置 {profile_id} 只用 {int(profile.input_channels)} 个：波束形成、声源方向、阵列回声抑制都不会生效。"
                if spare >= 4 else "")
        except Exception as exc:
            # "No input device matching 'MicArray'" on its own reads like a
            # configuration mistake, and the first thing anyone then checks is
            # the name they typed. Most of the time the array simply is not
            # enumerating, so say what the machine can actually see.
            with self.lock:
                self.error = f"input device preflight failed: {exc}；本机当前可用输入设备：{available_inputs()}"
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
            # Cheap: a memcpy on the callback thread. The embedding happens on the
            # worker below, never here.
            if self.audio_ring is not None:
                self.audio_ring.append(resample_mono(output.target_audio, frame.sample_rate_hz, 16000), frame.stamp_ms)
            if output.vad_active:
                self._speaker_context = (frame.stamp_ms, output.tracks[0].azimuth_deg if output.tracks else None, echo)
                # How much of the recent audio was actually speech. A window
                # that is mostly room tells the tracker about the room: with a
                # stale context and no check, a quiet room produced two dozen
                # "speaker" windows and kept them under whichever name was
                # matched first.
                self._speech_frames.append((frame.stamp_ms, int(frame.samples.shape[0] * 1000 / frame.sample_rate_hz)))
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
            if self.speaker_tracker is not None:
                self._speaker_stop.clear()
                self._speaker_worker = threading.Thread(target=self._track_speakers, daemon=True)
                self._speaker_worker.start()
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
                self.error = open_failure_message(exc, device_arg, profile.input_channels)
            self._teardown()

    # A window has to be mostly somebody talking. Measured: 1.5 s windows of
    # the owner's own voice score 0.75-0.90 against his pool, 0.6 s windows
    # anywhere from 0.11 to 0.71 — and a window of room noise scores against
    # whatever it likes.
    SPEECH_IN_WINDOW_MS = 800

    def _speech_ms_within(self, start_ms, end_ms):
        return sum(length for stamp, length in list(self._speech_frames) if start_ms <= stamp <= end_ms)

    def _track_speakers(self):
        """One embedding per hop of speech, off the audio callback."""
        while not self._speaker_stop.wait(.05):
            tracker, context = self.speaker_tracker, self._speaker_context
            if tracker is None or context is None:
                continue
            stamp_ms, azimuth, echo = context
            if not tracker.due(stamp_ms):
                continue
            if now_ms() - stamp_ms > max(400, tracker.hop_ms):
                # Speech stopped. The context is the last thing that was said,
                # not what the microphone is hearing now.
                continue
            if self._speech_ms_within(stamp_ms - tracker.window_ms, stamp_ms) < self.SPEECH_IN_WINDOW_MS:
                continue
            window, end_ms = self.audio_ring.window(tracker.window_ms)
            if window.size:
                try:
                    track = tracker.observe(window, end_ms or stamp_ms, azimuth, echo)
                    if track is not None and self.reference_resolver is not None and not track.reference_path:
                        # Capture a reference as they talk. Nobody should have to
                        # enrol before the robot can follow them in a crowd.
                        overlap = float((self.last_track or {}).get("overlap_probability") or 0)
                        self.reference_resolver.capture(track, {"overlap": overlap, "echo": echo}, write_wav)
                except Exception as exc:
                    with self.lock:
                        self.speaker_error = str(exc)

    def _finalize_utterance(self, utterance, stream, track, context):
        context = dict(context)
        generation = context.pop("_capture_generation", self._capture_generation)
        asr_worker = self._asr_worker
        if generation != self._capture_generation:
            return
        self._last_utterance_key = (utterance.started_ms, utterance.ended_ms)
        self._last_utterance_audio = utterance.samples.copy()
        self._last_utterance_rate = utterance.sample_rate_hz
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
            if (isinstance(self._asr_adapter, SherpaSenseVoiceAsrAdapter) and stream is not None
                    and not offline_pass_is_corroborated(streamed_final)):
                return
            asr_worker.submit(utterance)

    def stop(self) -> None:
        """A stop from outside also withdraws the standing capture request."""
        with self.lock:
            self.desired = None
        self._teardown()

    def _teardown(self) -> None:
        self._capture_generation += 1
        self._speaker_stop.set()
        worker, self._speaker_worker = self._speaker_worker, None
        if worker is not None:
            worker.join(timeout=1.0)
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
                "capture_warning": self.capture_warning,
                "capture_request": self.desired,
                "capture_retries": self.retry_count,
                "capture_reopens": self.reopen_count,
                "last_track": self.last_track,
                "speakers": self.speaker_tracker.state(now_ms()) if self.speaker_tracker else
                            {"speakers": [], "active": [], "reason": "no_speaker_model"},
                "attention_target": self.attention_target.current(now_ms()),
                # Who is registered, under what name, and how much voice each
                # pool actually holds — the console cannot show "陌生人3 说了…"
                # without it, and pool sizes are the first thing to look at
                # when somebody is not being recognised.
                "voice_registry": {
                    key: {"display_name": self.speaker_profiles.display_name(key),
                          "role": profile.get("role"),
                          "pool_size": len(profile.get("embeddings") or ([profile["embedding"]] if profile.get("embedding") else [])),
                          "anchor_groups": int(profile.get("anchor_groups") or 1),
                          "aliases": profile.get("aliases", []),
                          "has_reference": bool(profile.get("reference_path"))}
                    for key, profile in (self.speaker_profiles.profiles.items() if self.speaker_profiles else {}.items())
                    if profile.get("role") != "alias"
                },
                "speaker_thresholds": dict(self.speaker_profiles.thresholds) if self.speaker_profiles else {},
                "speaker_decision": dict(self.speaker_profiles.last_decision) if self.speaker_profiles else {},
                "reference_bank": {"session": dict(self.reference_resolver.session) if self.reference_resolver else {},
                                   "reason": self.reference_resolver.last_reason if self.reference_resolver else "none"},
                "last_vad": self.last_vad,
                "last_speaker": self.last_speaker,
                "last_event": self.last_event,
                "last_transcript": self.last_transcript,
                "active_utterance": self._segmenter.snapshot() if self._segmenter else None,
                "transcripts": [{key: value for key, value in item.items() if key != "speaker_vector"} for item in self.transcripts] if compact else list(self.transcripts),
                "playback": self.playback.state(),
                "last_streaming_transcript": self.last_streaming_transcript,
                "asr_enabled": bool(
                    os.environ.get("VOICE_ASR_SENSEVOICE_MODEL", "").strip()
                    or os.environ.get("VOICE_ASR_SHERPA_MODEL", "").strip()
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
                identity, role, similarity=self.speaker_profiles.remember_stranger(data["speaker_vector"])
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
            elif parsed.path == "/api/attention-target":
                payload=json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))) or "{}")
                resolver=monitor.reference_resolver
                speaker=str(payload.get("speaker_id") or "");person=payload.get("person_id")
                reference=resolver.resolve(speaker, person) if resolver else None
                if not reference:
                    # No reference means no target-speaker ASR. Refusing is the
                    # honest answer; conditioning on somebody else's voice would
                    # produce a confident transcript of the wrong person.
                    self._send_json({"accepted": False, "reason": resolver.last_reason if resolver else "no_resolver",
                                     "speaker_id": speaker})
                    return
                target=monitor.attention_target.set(dict(payload, reference_id=reference), now_ms())
                self._send_json({"accepted": True, "reference_used": reference, "target": target})
            elif parsed.path == "/api/speaker-thresholds":
                # Tunable while it runs: the bars that decide whether a voice is
                # somebody. Nothing here is calibrated on real-person ROC data,
                # so the field has to be able to move them and watch.
                payload=json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))) or "{}")
                if monitor.speaker_profiles is None:
                    self._send_json({"error":"no speaker model loaded"})
                else:
                    self._send_json({"thresholds":monitor.speaker_profiles.tune(**{
                        key:payload.get(key) for key in ("threshold","raw_floor","z_floor","margin_floor")
                        if payload.get(key) is not None})})
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
                    # This voice was auto-registered as a stranger and the face
                    # has now said who it is: the cluster belongs in that
                    # person's pool instead of competing with it.
                    cluster=str(payload.get("cluster_id") or "")
                    merged=bool(cluster and accepted and monitor.speaker_profiles.merge_cluster(cluster,str(payload.get("person_id"))))
                    self._send_json({"accepted":accepted,"cluster_merged":merged})
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


def refresh_audio_devices() -> bool:
    """Ask CoreAudio again instead of trusting PortAudio's cached list.

    PortAudio enumerates devices once, at initialisation. An array unplugged
    after that still appears in query_devices() while opening it fails with
    -9986 (paInternalError) — which reads like a driver bug and is in fact an
    empty USB port. Re-initialising is the only way to see what is there now.
    """
    try:
        import sounddevice as sd
        sd._terminate()
        sd._initialize()
        return True
    except Exception:
        return False


def input_device_channels(device_arg) -> int | None:
    """Input channels the device has right now, or None if it is not there."""
    try:
        import sounddevice as sd
        return int(sd.query_devices(device_arg, "input")["max_input_channels"])
    except Exception:
        return None


def open_failure_message(exc: Exception, device_arg, required_channels: int) -> str:
    """Say whether the port is empty or the device refused to open."""
    refresh_audio_devices()
    channels = input_device_channels(device_arg)
    label = "默认输入" if device_arg is None else f"“{device_arg}”"
    if channels is None:
        return (f"{label} 已经不在系统里了：设备在启动时还能枚举，打开采集流时已经消失，"
                f"通常是 USB 掉线（换根线、直插机器、不要串接总线供电的 HUB）。"
                f"原始错误：{exc}；当前可见输入设备：{available_inputs()}。"
                f"采集请求已保留，设备回来后会自动恢复。")
    if channels < int(required_channels):
        return (f"{label} 现在只有 {channels} 个输入通道，所选配置需要 {int(required_channels)} 个："
                f"设备可能以另一种模式重新枚举了。原始错误：{exc}")
    return (f"{label} 在系统里（{channels} 通道）但采集流打不开：可能被别的程序占用，或采样率/通道数不被接受。"
            f"原始错误：{exc}。采集请求已保留，会自动重试。")


def available_inputs() -> str:
    """Input devices this machine can actually see, for the preflight message."""
    try:
        devices = [f"[{item['index']}] {item['name']}({item['max_input_channels']}ch)"
                   for item in list_sounddevice_devices() if 'name' in item]
    except Exception as exc:
        return f"无法枚举（{exc}）"
    return "、".join(devices) if devices else "一个都没有"


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
