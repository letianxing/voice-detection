from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .doa import DoaEstimate, estimate_azimuth_gcc
from .io import read_wav
from .profiles import get_profile
from .types import MicrophoneProfile


@dataclass(frozen=True)
class ChannelMetric:
    channel_index: int
    rms_dbfs: float
    peak_dbfs: float
    silent: bool
    clipping: bool


@dataclass(frozen=True)
class WavCalibrationReport:
    wav_path: str
    sample_rate_hz: int
    channel_count: int
    frame_count: int
    duration_s: float
    profile_id: str | None
    expected_sample_rate_hz: int | None
    expected_channels: int | None
    channel_metrics: tuple[ChannelMetric, ...]
    correlation_matrix: tuple[tuple[float, ...], ...]
    doa_estimate: DoaEstimate | None
    warnings: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


def analyze_wav(path: str | Path, profile_id: str | None = None) -> WavCalibrationReport:
    frame = read_wav(path)
    samples = np.asarray(frame.samples, dtype=np.float32)
    profile = get_profile(profile_id) if profile_id else None

    metrics = tuple(_channel_metric(index, samples[:, index]) for index in range(samples.shape[1]))
    warnings = list(_profile_warnings(samples, frame.sample_rate_hz, profile))
    correlation = _correlation_matrix(samples)
    doa_estimate = _estimate_doa_if_possible(samples, frame.sample_rate_hz, profile, warnings)

    duration_s = float(samples.shape[0] / max(frame.sample_rate_hz, 1))
    return WavCalibrationReport(
        wav_path=str(Path(path).expanduser()),
        sample_rate_hz=frame.sample_rate_hz,
        channel_count=int(samples.shape[1]),
        frame_count=int(samples.shape[0]),
        duration_s=duration_s,
        profile_id=profile.id if profile else None,
        expected_sample_rate_hz=profile.sample_rate_hz if profile else None,
        expected_channels=profile.input_channels if profile else None,
        channel_metrics=metrics,
        correlation_matrix=correlation,
        doa_estimate=doa_estimate,
        warnings=tuple(warnings),
    )


def _channel_metric(index: int, samples: np.ndarray) -> ChannelMetric:
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return ChannelMetric(
        channel_index=index,
        rms_dbfs=_dbfs(rms),
        peak_dbfs=_dbfs(peak),
        silent=peak < 1e-4,
        clipping=peak >= 0.98,
    )


def _profile_warnings(samples: np.ndarray, sample_rate_hz: int, profile: MicrophoneProfile | None) -> tuple[str, ...]:
    if profile is None:
        return ()
    warnings: list[str] = []
    if sample_rate_hz != profile.sample_rate_hz:
        warnings.append(
            f"sample_rate_mismatch: wav={sample_rate_hz}, profile={profile.sample_rate_hz}"
        )
    if samples.shape[1] != profile.input_channels:
        warnings.append(f"channel_count_mismatch: wav={samples.shape[1]}, profile={profile.input_channels}")
    if profile.supports_doa and profile.doa_channel_indices:
        max_index = max(profile.doa_channel_indices)
        if samples.shape[1] <= max_index:
            warnings.append(f"doa_channels_missing: need channel index {max_index}")
    return tuple(warnings)


def _correlation_matrix(samples: np.ndarray) -> tuple[tuple[float, ...], ...]:
    if samples.shape[1] == 1:
        return ((1.0,),)
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    std = np.std(centered, axis=0)
    if np.any(std < 1e-8):
        return tuple(tuple(1.0 if row == col else 0.0 for col in range(samples.shape[1])) for row in range(samples.shape[1]))
    matrix = np.corrcoef(centered.T)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _estimate_doa_if_possible(
    samples: np.ndarray,
    sample_rate_hz: int,
    profile: MicrophoneProfile | None,
    warnings: list[str],
) -> DoaEstimate | None:
    if profile is None or not profile.supports_doa or not profile.doa_channel_indices:
        return None
    if samples.shape[1] <= max(profile.doa_channel_indices):
        return None
    try:
        return estimate_azimuth_gcc(
            samples=samples,
            sample_rate_hz=sample_rate_hz,
            mic_positions_m=profile.mic_positions_m,
            channel_indices=profile.doa_channel_indices,
        )
    except Exception as exc:
        warnings.append(f"doa_estimate_failed: {exc}")
        return None


def _dbfs(value: float) -> float:
    if value <= 0.0 or not math.isfinite(value):
        return -120.0
    return max(-120.0, float(20.0 * math.log10(value)))
