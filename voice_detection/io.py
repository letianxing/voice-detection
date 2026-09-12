from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .frontend import now_ms
from .types import AudioFrame


def read_wav(path: str | Path) -> AudioFrame:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate_hz = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError("only 16-bit PCM wav files are supported")
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    data = data.reshape((-1, channels))
    return AudioFrame(samples=data, sample_rate_hz=sample_rate_hz, stamp_ms=now_ms())


def write_wav(path: str | Path, frame: AudioFrame) -> None:
    data = np.asarray(frame.samples, dtype=np.float32)
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(int(data.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(frame.sample_rate_hz)
        handle.writeframes(pcm.tobytes())


def record_sounddevice(
    device: str | int | None,
    channels: int,
    sample_rate_hz: int,
    seconds: float,
) -> AudioFrame:
    if seconds <= 0.0:
        raise ValueError("recording seconds must be greater than zero")
    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError(f"sounddevice unavailable: {exc}") from exc
    device_arg = None if device in {None, "default"} else device
    frames = max(1, int(round(sample_rate_hz * seconds)))
    samples = sd.rec(
        frames,
        samplerate=sample_rate_hz,
        channels=channels,
        dtype="float32",
        device=device_arg,
        blocking=True,
    )
    return AudioFrame(
        samples=np.asarray(samples, dtype=np.float32).copy(),
        sample_rate_hz=sample_rate_hz,
        stamp_ms=now_ms(),
    )


def list_sounddevice_devices() -> list[dict[str, object]]:
    try:
        import sounddevice as sd
    except Exception as exc:
        return [{"error": f"sounddevice unavailable: {exc}"}]
    devices = sd.query_devices()
    return [
        {
            "index": index,
            "name": str(device["name"]),
            "max_input_channels": int(device["max_input_channels"]),
            "default_samplerate": float(device["default_samplerate"]),
            "hostapi": int(device["hostapi"]),
        }
        for index, device in enumerate(devices)
        if int(device["max_input_channels"]) > 0
    ]
