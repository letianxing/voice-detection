from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class MicrophoneProfile:
    id: str
    label: str
    sample_rate_hz: int
    input_channels: int
    doa_channel_indices: tuple[int, ...] = ()
    mic_positions_m: tuple[tuple[float, float, float], ...] = ()
    preferred_block_ms: int = 20
    supports_doa: bool = False
    supports_raw_multichannel: bool = False
    hardware_beam_channel_index: Optional[int] = None
    reference_channel_index: Optional[int] = None
    notes: str = ""


@dataclass(frozen=True)
class AudioFrame:
    samples: np.ndarray
    sample_rate_hz: int
    stamp_ms: int

    def __post_init__(self) -> None:
        if self.samples.ndim == 1:
            object.__setattr__(self, "samples", self.samples.reshape((-1, 1)))
        if self.samples.ndim != 2:
            raise ValueError("AudioFrame.samples must be shaped [frames, channels]")


@dataclass(frozen=True)
class VoiceActivity:
    active: bool
    probability: float
    rms_dbfs: float
    noise_floor_dbfs: float
    snr_db: float


@dataclass(frozen=True)
class AudioFeatures:
    zcr: float
    rms: float
    pitch: float = 0.0
    hnr: float = 0.0
    mfcc: tuple[float, ...] = field(default_factory=lambda: tuple(0.0 for _ in range(12)))


@dataclass(frozen=True)
class SpeechTranscript:
    track_id: str
    text: str
    is_final: bool
    language: str = "unknown"
    clarity: float = 0.0
    confidence: float = 0.0
    started_ms: Optional[int] = None
    ended_ms: Optional[int] = None
    emitted_ms: Optional[int] = None


@dataclass(frozen=True)
class AcousticTrack:
    track_id: str
    stamp_ms: int
    voice_activity: bool
    speech_probability: float
    clarity: float
    azimuth_deg: Optional[float] = None
    elevation_deg: Optional[float] = None
    distance_m: Optional[float] = None
    overlap_probability: float = 0.0
    self_echo_probability: float = 0.0
    speaker_label: str = "unknown"
    speaker_similarity: float = 0.0
    target_speaker_probability: float = 1.0
    tse_enabled: bool = False
    tse_healthy: bool = True
    tse_latency_ms: float = 0.0
    target_speech_rejected: bool = False
    speaker_embedding: tuple[float, ...] | None = None
    features: AudioFeatures | None = None
    transcript: SpeechTranscript | None = None
    raw_echo_probability: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["has_azimuth"] = self.azimuth_deg is not None
        data["has_elevation"] = self.elevation_deg is not None
        data["has_distance"] = self.distance_m is not None
        return data


@dataclass(frozen=True)
class FrontendOutput:
    stamp_ms: int
    tracks: tuple[AcousticTrack, ...]
    target_audio: np.ndarray
    target_track_id: str = ""
    vad_active: bool = False
    vad_probability: float = 0.0
    rms_dbfs: float = -120.0
    noise_floor_dbfs: float = -120.0
    snr_db: float = 0.0
    agc_gain_db: float = 0.0
