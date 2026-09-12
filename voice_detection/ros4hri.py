from __future__ import annotations

from dataclasses import asdict

from .types import AcousticTrack


def ros4hri_topics_for_voice(voice_id: str) -> dict[str, str]:
    prefix = f"/humans/voices/{voice_id}"
    return {
        "audio": f"{prefix}/audio",
        "features": f"{prefix}/features",
        "is_speaking": f"{prefix}/is_speaking",
        "speech": f"{prefix}/speech",
    }


def acoustic_track_to_ros4hri_json(track: AcousticTrack) -> dict[str, object]:
    features = asdict(track.features) if track.features is not None else None
    speech = None
    if track.transcript is not None:
        speech = {
            "incremental": "" if track.transcript.is_final else track.transcript.text,
            "final": track.transcript.text if track.transcript.is_final else "",
            "confidence": track.transcript.confidence,
            "locale": track.transcript.language.replace("-", "_"),
        }
    return {
        "tracked_topic": "/humans/voices/tracked",
        "voice_id": track.track_id,
        "voice_topics": ros4hri_topics_for_voice(track.track_id),
        "is_speaking": track.voice_activity,
        "features": features,
        "speech": speech,
        "engineering_track": track.to_json_dict(),
    }


def acoustic_track_to_attention_json(track: AcousticTrack) -> dict[str, object]:
    return {
        "track_id": track.track_id,
        "stamp_ms": track.stamp_ms,
        "voice_activity": track.voice_activity,
        "speech_probability": track.speech_probability,
        "clarity": track.clarity,
        "azimuth_deg": track.azimuth_deg,
        "elevation_deg": track.elevation_deg,
        "distance_m": track.distance_m,
        "overlap_probability": track.overlap_probability,
        "self_echo_probability": track.self_echo_probability,
        "speaker_label": track.speaker_label,
        "speaker_similarity": track.speaker_similarity,
    }
