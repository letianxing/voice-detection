from __future__ import annotations

from typing import Any, Mapping

from .frontend import now_ms
from .types import AcousticTrack, SpeechTranscript


def acoustic_track_from_audio_msg(payload: Mapping[str, Any], stamp_ms: int | None = None) -> AcousticTrack:
    """Map pacific-rim voice_service/msg/AudioMsg into the current track model.

    AudioMsg is a semantic/script-level message, not raw microphone audio. It is
    useful for compatibility tests and replay, but it cannot provide VAD energy,
    self-echo probability, overlap, or enhanced target audio.
    """

    text = str(payload.get("asr_text", "") or "").strip()
    user_id = str(payload.get("user_id", "") or "").strip()
    role = str(payload.get("role", "unknown") or "unknown").strip()
    track_id = str(payload.get("internal_id", "") or payload.get("voice_id", "") or user_id or "voice_legacy")
    stamp = stamp_ms if stamp_ms is not None else now_ms()
    transcript = None
    if text:
        transcript = SpeechTranscript(
            track_id=track_id,
            text=text,
            is_final=True,
            language=str(payload.get("language", "unknown") or "unknown"),
            clarity=0.5,
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            emitted_ms=stamp,
        )
    return AcousticTrack(
        track_id=track_id,
        stamp_ms=stamp,
        voice_activity=bool(text),
        speech_probability=1.0 if text else 0.0,
        clarity=0.5 if text else 0.0,
        azimuth_deg=_optional_float(payload.get("angle")),
        speaker_label=user_id or role or "unknown",
        transcript=transcript,
    )


def audio_msg_from_acoustic_track(track: AcousticTrack) -> dict[str, Any]:
    text = track.transcript.text if track.transcript is not None else ""
    return {
        "user_id": track.speaker_label if track.speaker_label != "unknown" else "",
        "role": "unknown",
        "asr_text": text,
        "emotion": "neutral",
        "angle": float(track.azimuth_deg) if track.azimuth_deg is not None else None,
        "angle_valid": track.azimuth_deg is not None,
        "speaker_vector": [],
    }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
