from __future__ import annotations

import argparse
import json
import time

from .audio_script import load_audio_script, replay_audio_script
from .calibration import analyze_wav
from .frontend import CocktailFrontend, now_ms
from .io import list_sounddevice_devices, read_wav, record_sounddevice, write_wav
from .profiles import get_profile, load_profiles
from .remote import run_live_remote, serve as serve_remote
from .ros4hri import acoustic_track_to_ros4hri_json
from .simulation import synthetic_array_frame
from .test_plan import write_asr_template
from .tts import speak_text
from .types import AudioFrame
from .pacific_rim_compat import acoustic_track_from_audio_msg


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("profiles")
    subparsers.add_parser("list-devices")

    offline = subparsers.add_parser("run-offline")
    offline.add_argument("--profile", default="mac_builtin")
    offline.add_argument("--wav", required=True)

    analyze = subparsers.add_parser("analyze-wav")
    analyze.add_argument("--wav", required=True)
    analyze.add_argument("--profile", default="")

    record = subparsers.add_parser("record-live")
    record.add_argument("--profile", default="sipeed_6_plus_1_usb_array")
    record.add_argument("--device", default="default")
    record.add_argument("--seconds", type=float, default=10.0)
    record.add_argument("--output", required=True)

    template = subparsers.add_parser("write-asr-template")
    template.add_argument("--output", required=True)
    template.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")

    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--profile", default="sipeed_6_plus_1_usb_array")
    simulate.add_argument("--azimuth", type=float, default=30.0)

    live = subparsers.add_parser("run-live")
    live.add_argument("--profile", default="mac_builtin")
    live.add_argument("--device", default="default")
    live.add_argument("--seconds", type=float, default=0.0)
    live.add_argument("--playback-reference-jsonl", default="")

    remote_server = subparsers.add_parser("run-remote-server")
    remote_server.add_argument("--host", default="0.0.0.0")
    remote_server.add_argument("--port", type=int, default=9097)

    remote_live = subparsers.add_parser("run-live-remote")
    remote_live.add_argument("--profile", default="sipeed_6_plus_1_usb_array")
    remote_live.add_argument("--device", default="default")
    remote_live.add_argument("--server-host", default="127.0.0.1")
    remote_live.add_argument("--server-port", type=int, default=9097)
    remote_live.add_argument("--seconds", type=float, default=0.0)
    remote_live.add_argument("--playback-reference-jsonl", default="")

    replay = subparsers.add_parser("replay-audio-script")
    replay.add_argument("--script", required=True)
    replay.add_argument("--loops", type=int, default=1)
    replay.add_argument("--no-wait", action="store_true")

    speak = subparsers.add_parser("speak")
    speak.add_argument("--text", required=True)
    speak.add_argument("--backend", default="auto")
    speak.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.command == "profiles":
        for profile in load_profiles().values():
            print(json.dumps(profile.__dict__, ensure_ascii=False, separators=(",", ":")))
    elif args.command == "list-devices":
        print(json.dumps(list_sounddevice_devices(), ensure_ascii=False, indent=2))
    elif args.command == "run-offline":
        profile = get_profile(args.profile)
        frame = read_wav(args.wav)
        emit_output(CocktailFrontend(profile).process(frame))
    elif args.command == "analyze-wav":
        profile_id = args.profile if args.profile else None
        print(analyze_wav(args.wav, profile_id=profile_id).to_json())
    elif args.command == "record-live":
        profile = get_profile(args.profile)
        frame = record_sounddevice(
            coerce_device(args.device),
            profile.input_channels,
            profile.sample_rate_hz,
            args.seconds,
        )
        write_wav(args.output, frame)
        print(analyze_wav(args.output, profile_id=profile.id).to_json())
    elif args.command == "write-asr-template":
        write_asr_template(args.output, fmt=args.format)
        print(json.dumps({"output": args.output, "format": args.format}, ensure_ascii=False))
    elif args.command == "simulate":
        profile = get_profile(args.profile)
        frame = synthetic_array_frame(profile, azimuth_deg=args.azimuth)
        emit_output(CocktailFrontend(profile).process(frame))
    elif args.command == "run-live":
        run_live(args.profile, args.device, args.seconds, args.playback_reference_jsonl)
    elif args.command == "run-remote-server":
        serve_remote(args.host, args.port)
    elif args.command == "run-live-remote":
        run_live_remote(
            profile_id=args.profile,
            device=args.device,
            server_host=args.server_host,
            server_port=args.server_port,
            seconds=args.seconds,
            playback_reference_jsonl=args.playback_reference_jsonl,
        )
    elif args.command == "replay-audio-script":
        replay_script(args.script, args.loops, args.no_wait)
    elif args.command == "speak":
        print(speak_text(args.text, backend=args.backend, dry_run=args.dry_run).to_json())


def emit_output(output) -> None:
    for track in output.tracks:
        print(json.dumps(acoustic_track_to_ros4hri_json(track), ensure_ascii=False, separators=(",", ":")))


def run_live(profile_id: str, device: str, seconds: float, playback_reference_jsonl: str = "") -> None:
    try:
        import sounddevice as sd
        import numpy as np
    except Exception as exc:
        raise SystemExit(f"sounddevice is required for live capture: {exc}")

    profile = get_profile(profile_id)
    frontend = CocktailFrontend(profile, vad_calibration_frames=15)
    reference_source = None
    if playback_reference_jsonl:
        from .aec import JsonlPlaybackReferenceSource

        reference_source = JsonlPlaybackReferenceSource(playback_reference_jsonl)
    blocksize = int(profile.sample_rate_hz * profile.preferred_block_ms / 1000.0)
    device_arg = None if device == "default" else coerce_device(device)
    started = time.time()

    def callback(indata, frames, callback_time, status):
        del frames, callback_time
        if status:
            print(json.dumps({"warning": str(status)}, ensure_ascii=False), flush=True)
        frame = AudioFrame(
            samples=np.asarray(indata, dtype=np.float32).copy(),
            sample_rate_hz=profile.sample_rate_hz,
            stamp_ms=now_ms(),
        )
        reference = reference_source.read_for(frame) if reference_source is not None else None
        output = frontend.process(frame, reference)
        for track in output.tracks:
            print(json.dumps(acoustic_track_to_ros4hri_json(track), ensure_ascii=False), flush=True)

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
        if reference_source is not None:
            reference_source.close()


def replay_script(path: str, loops: int, no_wait: bool) -> None:
    plan = load_audio_script(path)
    if plan.message_type != "voice_service/msg/AudioMsg":
        raise SystemExit(f"unsupported replay message type: {plan.message_type}")

    def emit(payload) -> None:
        track = acoustic_track_from_audio_msg(payload)
        print(json.dumps(acoustic_track_to_ros4hri_json(track), ensure_ascii=False), flush=True)

    sleeper = (lambda _seconds: None) if no_wait else time.sleep
    replay_audio_script(plan, emit, loops=loops, sleep=sleeper)


def coerce_device(value: str):
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
