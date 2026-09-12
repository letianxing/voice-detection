from __future__ import annotations

import argparse
import base64
import json
import queue
import socket
import socketserver
import threading
import time
from typing import Any

import numpy as np

from .frontend import CocktailFrontend, now_ms
from .profiles import get_profile
from .ros4hri import acoustic_track_to_ros4hri_json
from .types import AudioFrame


Packet = dict[str, Any]


def encode_audio_frame(
    frame: AudioFrame,
    profile_id: str,
    playback_reference: np.ndarray | None = None,
) -> Packet:
    samples = np.clip(frame.samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2", copy=False)
    packet = {
        "type": "audio_frame",
        "profile_id": profile_id,
        "sample_rate_hz": frame.sample_rate_hz,
        "channels": int(frame.samples.shape[1]),
        "frames": int(frame.samples.shape[0]),
        "stamp_ms": frame.stamp_ms,
        "dtype": "s16le",
        "pcm_b64": base64.b64encode(pcm.tobytes()).decode("ascii"),
    }
    if playback_reference is not None:
        reference = np.clip(np.asarray(playback_reference, dtype=np.float32).reshape(-1), -1.0, 1.0)
        reference_pcm = (reference * 32767.0).astype("<i2", copy=False)
        packet["playback_reference_frames"] = int(reference.size)
        packet["playback_reference_pcm_b64"] = base64.b64encode(reference_pcm.tobytes()).decode("ascii")
    return packet


def decode_audio_frame(packet: Packet) -> AudioFrame:
    if packet.get("type") != "audio_frame":
        raise ValueError(f"unsupported packet type: {packet.get('type')}")
    if packet.get("dtype") != "s16le":
        raise ValueError(f"unsupported audio dtype: {packet.get('dtype')}")
    channels = int(packet["channels"])
    frames = int(packet["frames"])
    raw = base64.b64decode(str(packet["pcm_b64"]))
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    samples = samples.reshape((frames, channels))
    return AudioFrame(
        samples=samples,
        sample_rate_hz=int(packet["sample_rate_hz"]),
        stamp_ms=int(packet["stamp_ms"]),
    )


def decode_playback_reference(packet: Packet) -> np.ndarray | None:
    encoded = packet.get("playback_reference_pcm_b64")
    if encoded is None:
        return None
    expected = int(packet.get("playback_reference_frames", 0))
    reference = np.frombuffer(base64.b64decode(str(encoded)), dtype="<i2").astype(np.float32) / 32768.0
    if expected != reference.size:
        raise ValueError("playback reference frame count does not match payload")
    return reference


def frontend_output_packet(output) -> Packet:
    return {
        "type": "frontend_output",
        "stamp_ms": output.stamp_ms,
        "target_track_id": output.target_track_id,
        "vad": {
            "active": output.vad_active,
            "probability": output.vad_probability,
            "rms_dbfs": output.rms_dbfs,
            "noise_floor_dbfs": output.noise_floor_dbfs,
            "snr_db": output.snr_db,
            "agc_gain_db": output.agc_gain_db,
        },
        "last_track": output.tracks[0].to_json_dict() if output.tracks else None,
        "tracks": [track.to_json_dict() for track in output.tracks],
        "ros4hri": [acoustic_track_to_ros4hri_json(track) for track in output.tracks],
    }


def send_packet(sock: socket.socket, packet: Packet) -> None:
    data = json.dumps(packet, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    sock.sendall(data)


def read_packet(file_obj) -> Packet | None:
    line = file_obj.readline()
    if not line:
        return None
    return json.loads(line.decode("utf-8"))


class AcousticRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        frontend: CocktailFrontend | None = None
        active_profile_id = ""
        while True:
            packet = read_packet(self.rfile)
            if packet is None:
                return
            profile_id = str(packet.get("profile_id", "mac_builtin"))
            if frontend is None or profile_id != active_profile_id:
                frontend = CocktailFrontend(get_profile(profile_id), vad_calibration_frames=15)
                active_profile_id = profile_id
            frame = decode_audio_frame(packet)
            output = frontend.process(frame, decode_playback_reference(packet))
            response = json.dumps(
                frontend_output_packet(output),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            self.wfile.write(response)
            self.wfile.flush()


class AcousticServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class RemoteAcousticClient:
    def __init__(self, host: str, port: int, timeout: float = 3.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.file = self.sock.makefile("rb")

    def close(self) -> None:
        try:
            self.file.close()
        finally:
            self.sock.close()

    def process(
        self,
        frame: AudioFrame,
        profile_id: str,
        playback_reference: np.ndarray | None = None,
    ) -> Packet:
        send_packet(self.sock, encode_audio_frame(frame, profile_id, playback_reference))
        response = read_packet(self.file)
        if response is None:
            raise ConnectionError("remote acoustic service closed the connection")
        return response


def serve(host: str, port: int) -> None:
    with AcousticServer((host, port), AcousticRequestHandler) as server:
        print(f"remote acoustic service: {host}:{port}", flush=True)
        server.serve_forever()


def run_live_remote(
    profile_id: str,
    device: str,
    server_host: str,
    server_port: int,
    seconds: float,
    playback_reference_jsonl: str = "",
) -> None:
    try:
        import sounddevice as sd
    except Exception as exc:
        raise SystemExit(f"sounddevice is required for live capture: {exc}")

    profile = get_profile(profile_id)
    blocksize = int(profile.sample_rate_hz * profile.preferred_block_ms / 1000.0)
    device_arg = None if device == "default" else coerce_device(device)
    reference_source = None
    if playback_reference_jsonl:
        from .aec import JsonlPlaybackReferenceSource

        reference_source = JsonlPlaybackReferenceSource(playback_reference_jsonl)
    frames: queue.Queue[tuple[AudioFrame, np.ndarray | None]] = queue.Queue(maxsize=8)
    stopped = threading.Event()

    def callback(indata, frame_count, callback_time, status):
        del frame_count, callback_time
        if status:
            print(json.dumps({"warning": str(status)}, ensure_ascii=False), flush=True)
        frame = AudioFrame(
            samples=np.asarray(indata, dtype=np.float32).copy(),
            sample_rate_hz=profile.sample_rate_hz,
            stamp_ms=now_ms(),
        )
        try:
            reference = reference_source.read_for(frame) if reference_source is not None else None
            frames.put_nowait((frame, reference))
        except queue.Full:
            print(json.dumps({"warning": "remote_frame_queue_full"}, ensure_ascii=False), flush=True)

    def worker() -> None:
        client = RemoteAcousticClient(server_host, server_port)
        try:
            while not stopped.is_set():
                try:
                    frame, reference = frames.get(timeout=0.2)
                except queue.Empty:
                    continue
                response = client.process(frame, profile_id, reference)
                print(json.dumps(response, ensure_ascii=False), flush=True)
        finally:
            client.close()

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    started = time.time()
    try:
        with sd.InputStream(
            device=device_arg,
            channels=profile.input_channels,
            samplerate=profile.sample_rate_hz,
            blocksize=blocksize,
            dtype="float32",
            callback=callback,
        ):
            while seconds <= 0.0 or time.time() - started < seconds:
                time.sleep(0.1)
    finally:
        stopped.set()
        worker_thread.join(timeout=1.0)
        if reference_source is not None:
            reference_source.close()


def coerce_device(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9097)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
