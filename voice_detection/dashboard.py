from __future__ import annotations

import argparse
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
from .asr import ExternalCommandAsrAdapter, NullAsrAdapter, SherpaZipformerAsrAdapter, VoskAsrAdapter
from .io import list_sounddevice_devices
from .profiles import get_profile, load_profiles
from .ros4hri import acoustic_track_to_ros4hri_json
from .types import AudioFrame
from .streaming_asr import AsrWorker, UtteranceSegmenter
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
      <div class="metric">Azimuth<strong id="az">-</strong></div>
      <div class="metric">Clarity<strong id="clarity">-</strong></div>
      <div class="metric">Overlap<strong id="overlap">-</strong></div>
      <div class="metric">Speaker<strong id="speaker">-</strong></div>
      <div class="metric">Similarity<strong id="similarity">-</strong></div>
      <div class="metric">ASR partial<strong id="partial">-</strong></div>
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
      const streaming = state.last_streaming_transcript || {};
      document.getElementById("partial").textContent = streaming.text || "-";
      log.textContent = JSON.stringify(state, null, 2);
    }
    document.getElementById("start").onclick = async () => {
      const params = new URLSearchParams({profile: profile.value, device: device.value, seconds: document.getElementById("seconds").value});
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
        self.last_transcript: dict[str, object] | None = None
        self.last_streaming_transcript: dict[str, object] | None = None
        self._streaming_asr = None
        self._streaming_adapter = None
        self._streaming_started_ms = 0
        self.asr_error = ""
        self.asr_backend = "none"
        self._sample_rate_hz = 48000
        self.robot_speaking = False
        self.input_device_info: dict[str, object] | None = None

    def start(self, profile_id: str, device: str, seconds: float) -> None:
        self.stop()
        try:
            import numpy as np
            import sounddevice as sd
        except Exception as exc:
            with self.lock:
                self.error = f"sounddevice is required for live capture: {exc}"
            return

        profile = get_profile(profile_id)
        self._sample_rate_hz = profile.sample_rate_hz
        frontend = CocktailFrontend(profile, vad_calibration_frames=15)
        model_path = os.environ.get(
            "VOICE_SPEAKER_MODEL",
            str(Path(__file__).resolve().parents[1] / "weights" / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"),
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
        self._asr_worker = AsrWorker(adapter, self._on_transcript, self._on_asr_error)
        self._streaming_asr = None
        # Pre-spawn and warm the Sherpa worker before the first spoken frame;
        # model loading must never block the PortAudio callback.
        if self._streaming_adapter is not None and isinstance(adapter, SherpaZipformerAsrAdapter):
            try:
                self._streaming_asr = self._streaming_adapter.create_stream("voice_mono", self._sample_rate_hz)
                self._streaming_asr.accept(np.zeros(blocksize if 'blocksize' in locals() else 960, dtype=np.float32), self._sample_rate_hz, now_ms())
            except Exception:
                self._streaming_asr = None
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

        def callback(indata, frames, callback_time, status):
            del frames, callback_time
            event = {"stamp_ms": now_ms(), "status": str(status) if status else ""}
            frame = AudioFrame(
                samples=np.asarray(indata, dtype=np.float32).copy(),
                sample_rate_hz=profile.sample_rate_hz,
                stamp_ms=event["stamp_ms"],
            )
            output = frontend.process(frame)
            track = output.tracks[0] if output.tracks else None
            if track is not None and self.robot_speaking:
                track = replace(track, self_echo_probability=1.0)
            utterance = self._segmenter.process(
                output.target_audio,
                frame.sample_rate_hz,
                frame.stamp_ms,
                bool(track and track.voice_activity and not self.robot_speaking),
                track.track_id if track else "",
            )
            if bool(track and track.voice_activity and not self.robot_speaking):
                if self._streaming_asr is None and self._streaming_adapter is not None:
                    self._streaming_started_ms = frame.stamp_ms
                    self._streaming_asr = self._streaming_adapter.create_stream(track.track_id if track else "voice_mono", frame.sample_rate_hz)
                    # Do not leave the previous utterance's partial text on
                    # screen while the next utterance is being decoded.
                    with self.lock:
                        self.last_streaming_transcript = {
                            "track_id": track.track_id if track else "voice_mono",
                            "text": "",
                            "is_final": False,
                            "started_ms": self._streaming_started_ms,
                            "emitted_ms": None,
                            "first_result_latency_ms": None,
                        }
                if self._streaming_asr is not None:
                    try:
                        partial = self._streaming_asr.accept(output.target_audio, frame.sample_rate_hz, frame.stamp_ms)
                    except Exception as exc:
                        partial = ""
                        with self.lock:
                            self.asr_error = f"streaming_asr: {exc}"
                    if partial:
                        with self.lock:
                            emitted_ms = now_ms()
                            self.last_streaming_transcript = {
                                "track_id": track.track_id if track else "voice_mono",
                                "text": partial,
                                "is_final": False,
                                "started_ms": self._streaming_started_ms,
                                "emitted_ms": emitted_ms,
                                "first_result_latency_ms": max(0, emitted_ms - self._streaming_started_ms),
                            }
            if utterance is not None:
                if self._streaming_asr is not None:
                    try:
                        streamed_final = self._streaming_asr.finish(utterance.ended_ms)
                    except Exception as exc:
                        streamed_final = None
                        with self.lock:
                            self.asr_error = f"streaming_asr_finish: {exc}"
                    self._streaming_asr = None
                    if streamed_final is not None:
                        with self.lock:
                            self.last_streaming_transcript = {
                                **asdict(streamed_final),
                                "first_result_latency_ms": max(
                                    0,
                                    int(self.last_streaming_transcript.get("first_result_latency_ms") or 0)
                                    if self.last_streaming_transcript
                                    else 0,
                                ),
                                "final_latency_ms": max(0, now_ms() - utterance.ended_ms),
                            }
                embedding = self.speaker_embedder.embed(utterance.samples, utterance.sample_rate_hz) if self.speaker_embedder else None
                if embedding:
                    speaker_id, speaker_role, similarity = self.speaker_profiles.match(embedding)
                    with self.lock:
                        self.last_speaker = {
                            "speaker_id": speaker_id,
                            "speaker_role": speaker_role,
                            "similarity": similarity,
                            "embedding": embedding,
                            "track_id": utterance.track_id,
                            "stamp_ms": utterance.ended_ms,
                        }
                        if track is not None:
                            track = replace(
                                track,
                                speaker_label=speaker_id,
                                speaker_similarity=float(similarity),
                                speaker_embedding=tuple(embedding),
                            )
                self._asr_worker.submit(utterance)
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

    def stop(self) -> None:
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
        if self._asr_worker is not None:
            self._asr_worker.close()
            self._asr_worker = None
        self._segmenter = None
        self._streaming_asr = None
        self._streaming_adapter = None
        with self.lock:
            self.running = False

    def state(self) -> dict[str, object]:
        with self.lock:
            return {
                "running": self.running,
                "error": self.error,
                "last_track": self.last_track,
                "last_vad": self.last_vad,
                "last_speaker": self.last_speaker,
                "last_event": self.last_event,
                "last_transcript": self.last_transcript,
                "last_streaming_transcript": self.last_streaming_transcript,
                "asr_enabled": bool(
                    os.environ.get("VOICE_ASR_COMMAND", "").strip()
                    or os.environ.get("VOICE_ASR_VOSK_MODEL", "").strip()
                ),
                "asr_error": self.asr_error,
                "asr_backend": self.asr_backend,
                "robot_speaking": self.robot_speaking,
                "input_device_info": self.input_device_info,
            }

    def set_robot_speaking(self, speaking: bool) -> None:
        with self.lock:
            self.robot_speaking = bool(speaking)

    def _on_transcript(self, transcript) -> None:
        with self.lock:
            self.last_transcript = asdict(transcript)
            self.asr_error = ""

    def _on_asr_error(self, error: str) -> None:
        with self.lock:
            self.asr_error = error


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
            elif parsed.path == "/api/state":
                self._send_json(monitor.state())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/api/start":
                monitor.start(
                    profile_id=query.get("profile", ["mac_builtin"])[0],
                    device=query.get("device", ["default"])[0],
                    seconds=float(query.get("seconds", ["0"])[0] or 0),
                )
                self._send_json(monitor.state())
            elif parsed.path == "/api/stop":
                monitor.stop()
                self._send_json(monitor.state())
            elif parsed.path == "/api/robot-speaking":
                monitor.set_robot_speaking(query.get("speaking", ["0"])[0] in {"1", "true", "True"})
                self._send_json(monitor.state())
            elif parsed.path == "/api/enroll-speaker":
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8") or "{}")
                speaker_id = str(payload.get("speaker_id") or "").strip()
                if not speaker_id or not monitor.last_speaker or not monitor.last_speaker.get("embedding"):
                    self._send_json({"success": False, "message": "speak first, then provide speaker_id"})
                    return
                monitor.speaker_profiles.enroll(
                    speaker_id,
                    monitor.last_speaker["embedding"],
                    str(payload.get("speaker_role") or "known"),
                )
                monitor.last_speaker["speaker_id"] = speaker_id
                monitor.last_speaker["speaker_role"] = str(payload.get("speaker_role") or "known")
                monitor.last_speaker["similarity"] = 1.0
                self._send_json({"success": True, "speaker_id": speaker_id})
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
    finally:
        monitor.stop()
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
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
