from __future__ import annotations

import time

import numpy as np

from .aec import LinearEchoCanceller
from .beamforming import delay_and_sum, mixdown
from .doa import estimate_azimuth_gcc
from .features import extract_features
from .noise import (
    AdaptiveNoiseSuppressor,
    estimate_clarity_from_snr,
    estimate_overlap_probability,
    estimate_self_echo_probability,
    remove_dc,
)
from .tracker import SourceTracker
from .types import AcousticTrack, AudioFrame, FrontendOutput, MicrophoneProfile
from .vad import EnergyVad


class CocktailFrontend:
    """Low-latency acoustic frontend for one dominant voice per audio block.

    The first implementation always provides VAD, clarity and target mixdown.
    With a calibrated array profile it adds GCC-PHAT DOA and delay-and-sum
    beamforming. Neural separation can be attached behind `FrontendOutput`
    without changing ROS4HRI or attention interfaces.
    """

    def __init__(self, profile: MicrophoneProfile, vad_calibration_frames: int = 0):
        self.profile = profile
        self.pre_vad = EnergyVad(calibration_frames=vad_calibration_frames)
        self.post_vad = EnergyVad(calibration_frames=vad_calibration_frames)
        self.noise_suppressor = AdaptiveNoiseSuppressor()
        self.echo_canceller = LinearEchoCanceller()
        self.tracker = SourceTracker()

    def process(self, frame: AudioFrame, playback_reference: np.ndarray | None = None) -> FrontendOutput:
        samples = remove_dc(frame.samples)
        # Keep the original, stable frontend as the default signal path.  In
        # particular, do not switch channels or alter gain independently for
        # every 20 ms block: both operations move the adaptive VAD noise floor
        # and change the waveform presented to ASR.
        raw_mono = mixdown(samples, self.profile.doa_channel_indices or None)
        pre_vad = self.pre_vad.process(raw_mono)

        azimuth_deg = None
        doa_confidence = 0.0
        if pre_vad.active and self.profile.supports_doa and self.profile.doa_channel_indices:
            doa = estimate_azimuth_gcc(
                samples=samples,
                sample_rate_hz=frame.sample_rate_hz,
                mic_positions_m=self.profile.mic_positions_m,
                channel_indices=self.profile.doa_channel_indices,
            )
            azimuth_deg = doa.azimuth_deg
            doa_confidence = doa.confidence

        if azimuth_deg is not None and doa_confidence >= 0.1:
            spatial_audio = delay_and_sum(
                samples=samples,
                sample_rate_hz=frame.sample_rate_hz,
                mic_positions_m=self.profile.mic_positions_m,
                channel_indices=self.profile.doa_channel_indices,
                azimuth_deg=azimuth_deg,
            )
        else:
            spatial_audio = raw_mono
        raw_echo = estimate_self_echo_probability(spatial_audio, playback_reference)
        echo_cancelled = self.echo_canceller.process(spatial_audio, playback_reference)
        self_echo = estimate_self_echo_probability(echo_cancelled, playback_reference)
        target_audio = self.noise_suppressor.process(echo_cancelled, pre_vad.active)
        vad = self.post_vad.process(target_audio)
        clarity = estimate_clarity_from_snr(vad.snr_db)

        track_id = self.tracker.assign(frame.stamp_ms, azimuth_deg) if vad.active else ""
        features = extract_features(target_audio, frame.sample_rate_hz)
        overlap = estimate_overlap_probability(samples) if vad.active else 0.0
        tracks = ()
        if vad.active:
            tracks = (
                AcousticTrack(
                    track_id=track_id,
                    stamp_ms=frame.stamp_ms,
                    voice_activity=vad.active,
                    speech_probability=vad.probability,
                    clarity=clarity,
                    azimuth_deg=azimuth_deg,
                    overlap_probability=overlap,
                    self_echo_probability=self_echo,
                    raw_echo_probability=raw_echo,
                    features=features,
                ),
            )
        return FrontendOutput(
            stamp_ms=frame.stamp_ms,
            tracks=tracks,
            target_audio=target_audio.astype(np.float32),
            target_track_id=track_id,
            vad_active=vad.active,
            vad_probability=vad.probability,
            rms_dbfs=vad.rms_dbfs,
            noise_floor_dbfs=vad.noise_floor_dbfs,
            snr_db=vad.snr_db,
            agc_gain_db=0.0,
        )


def now_ms() -> int:
    return int(time.time() * 1000)
