from __future__ import annotations

from dataclasses import dataclass

from .doa import normalize_angle_deg


@dataclass
class _TrackState:
    track_id: str
    azimuth_deg: float | None
    last_seen_ms: int


class SourceTracker:
    def __init__(self, max_angle_delta_deg: float = 25.0, max_age_ms: int = 1200):
        self.max_angle_delta_deg = max_angle_delta_deg
        self.max_age_ms = max_age_ms
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    def assign(self, stamp_ms: int, azimuth_deg: float | None) -> str:
        self._expire(stamp_ms)
        if azimuth_deg is None:
            return "voice_mono"
        azimuth = normalize_angle_deg(azimuth_deg)
        best: _TrackState | None = None
        best_delta = float("inf")
        for track in self._tracks:
            if track.azimuth_deg is None:
                continue
            delta = angular_distance_deg(azimuth, track.azimuth_deg)
            if delta < best_delta:
                best = track
                best_delta = delta
        if best is not None and best_delta <= self.max_angle_delta_deg:
            best.azimuth_deg = azimuth
            best.last_seen_ms = stamp_ms
            return best.track_id
        track = _TrackState(track_id=f"voice_{self._next_id}", azimuth_deg=azimuth, last_seen_ms=stamp_ms)
        self._next_id += 1
        self._tracks.append(track)
        return track.track_id

    def tracked_ids(self, stamp_ms: int) -> tuple[str, ...]:
        self._expire(stamp_ms)
        return tuple(track.track_id for track in self._tracks)

    def _expire(self, stamp_ms: int) -> None:
        self._tracks = [
            track for track in self._tracks
            if stamp_ms - track.last_seen_ms <= self.max_age_ms
        ]


def angular_distance_deg(a: float, b: float) -> float:
    delta = abs((a - b + 180.0) % 360.0 - 180.0)
    return min(delta, 360.0 - delta)

