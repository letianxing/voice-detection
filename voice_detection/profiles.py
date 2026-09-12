from __future__ import annotations

import json
from pathlib import Path

from .types import MicrophoneProfile


DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "config" / "microphones.json"


def load_profiles(path: str | Path = DEFAULT_PROFILE_PATH) -> dict[str, MicrophoneProfile]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = {}
    for item in raw["profiles"]:
        profile = MicrophoneProfile(
            id=item["id"],
            label=item["label"],
            sample_rate_hz=int(item["sample_rate_hz"]),
            input_channels=int(item["input_channels"]),
            doa_channel_indices=tuple(int(index) for index in item.get("doa_channel_indices", [])),
            mic_positions_m=tuple(tuple(float(v) for v in pos) for pos in item.get("mic_positions_m", [])),
            preferred_block_ms=int(item.get("preferred_block_ms", 20)),
            supports_doa=bool(item.get("supports_doa", False)),
            supports_raw_multichannel=bool(item.get("supports_raw_multichannel", False)),
            hardware_beam_channel_index=optional_int(item.get("hardware_beam_channel_index")),
            reference_channel_index=optional_int(item.get("reference_channel_index")),
            notes=item.get("notes", ""),
        )
        profiles[profile.id] = profile
    return profiles


def get_profile(profile_id: str, path: str | Path = DEFAULT_PROFILE_PATH) -> MicrophoneProfile:
    profiles = load_profiles(path)
    try:
        return profiles[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown microphone profile: {profile_id}") from exc


def optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)
