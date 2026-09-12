from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .types import AudioFrame


def encode_audio_frame_payload(
    samples: np.ndarray,
    sample_rate_hz: int,
    frame_id: str,
    *,
    stamp_sec: int = 0,
    stamp_nanosec: int = 0,
    encoding: str = "PCM",
    interleaved: bool = True,
) -> bytes:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        channels = 1
        flat = data
    elif data.ndim == 2:
        channels = int(data.shape[1])
        flat = data.reshape(-1) if interleaved else data.T.reshape(-1)
    else:
        raise ValueError("audio samples must be shaped [frames] or [frames, channels]")
    payload = {
        "header": {
            "frame_id": str(frame_id),
            "stamp_sec": int(stamp_sec),
            "stamp_nanosec": int(stamp_nanosec),
        },
        "sample_rate": int(sample_rate_hz),
        "channels": int(channels),
        "encoding": str(encoding),
        "interleaved": bool(interleaved),
        "data": np.clip(flat, -1.0, 1.0).astype(np.float32).tolist(),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_audio_frame_payload(payload: bytes | str | Mapping[str, Any]) -> AudioFrame:
    raw = _load_payload(payload)
    channels = max(1, int(raw.get("channels", 1) or 1))
    sample_rate_hz = int(raw.get("sample_rate", raw.get("sample_rate_hz", 0)) or 0)
    if sample_rate_hz <= 0:
        raise ValueError("audio frame sample_rate is required")
    data = np.asarray(raw.get("data") or [], dtype=np.float32)
    if data.size % channels != 0:
        raise ValueError("audio frame data length must be divisible by channels")
    if bool(raw.get("interleaved", True)):
        samples = data.reshape((-1, channels))
    else:
        samples = data.reshape((channels, -1)).T
    stamp_ms = _stamp_ms(raw) or int(time.time() * 1000)
    return AudioFrame(samples=np.clip(samples, -1.0, 1.0).astype(np.float32), sample_rate_hz=sample_rate_hz, stamp_ms=stamp_ms)


def pcm16_to_float32(pcm_int16: np.ndarray) -> np.ndarray:
    pcm = np.asarray(pcm_int16)
    if pcm.size == 0:
        return np.zeros(0, dtype=np.float32)
    if pcm.dtype != np.int16:
        pcm = pcm.astype(np.int16)
    return (pcm.astype(np.float32) / 32767.0).clip(-1.0, 1.0)


def float32_to_pcm16(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.size == 0:
        return np.zeros(0, dtype=np.int16)
    return (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)


def load_audio_frame_payload(path: str | Path) -> AudioFrame:
    return decode_audio_frame_payload(Path(path).expanduser().read_bytes())


def _load_payload(payload: bytes | str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, Mapping):
        raise ValueError("audio frame payload must be a JSON object")
    return data


def _stamp_ms(payload: Mapping[str, Any]) -> int | None:
    header = payload.get("header")
    if not isinstance(header, Mapping):
        return None
    sec = header.get("stamp_sec")
    nanosec = header.get("stamp_nanosec")
    if sec is None and nanosec is None:
        return None
    return int(sec or 0) * 1000 + int(int(nanosec or 0) / 1_000_000)
