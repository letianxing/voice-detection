from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Mapping


DEFAULT_MESSAGE_TYPE = "voice_service/msg/AudioMsg"
DEFAULT_TOPIC_NAME = "/voice_msg_speaker"
_CONTROL_KEYS = {"count", "message", "payload", "period_sec", "rate_hz"}


@dataclass(frozen=True)
class AudioScriptStep:
    payload: Mapping[str, Any]
    period_sec: float = 1.0
    count: int = 1


@dataclass(frozen=True)
class AudioScriptPlan:
    topic_name: str
    message_type: str
    loop: bool
    steps: tuple[AudioScriptStep, ...]


def load_audio_script(path: str | Path) -> AudioScriptPlan:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("audio script must contain a JSON object")
    return load_audio_script_data(raw)


def load_audio_script_data(raw: Mapping[str, Any]) -> AudioScriptPlan:
    steps = raw.get("steps") or raw.get("script") or raw.get("messages")
    if not isinstance(steps, list) or not steps:
        raise ValueError("audio script must define a non-empty steps list")
    return AudioScriptPlan(
        topic_name=_clean_string(raw.get("topic_name") or raw.get("topic") or DEFAULT_TOPIC_NAME),
        message_type=_clean_string(raw.get("message_type") or DEFAULT_MESSAGE_TYPE),
        loop=bool(raw.get("loop", False)),
        steps=tuple(_load_step(index, step) for index, step in enumerate(steps)),
    )


def iter_script_payloads(plan: AudioScriptPlan, loops: int = 1) -> Iterator[tuple[Mapping[str, Any], float]]:
    if loops <= 0:
        raise ValueError("loops must be greater than zero")
    for _ in range(loops):
        for step in plan.steps:
            for _ in range(step.count):
                yield dict(step.payload), step.period_sec


def replay_audio_script(
    plan: AudioScriptPlan,
    emit: Callable[[Mapping[str, Any]], None],
    *,
    loops: int = 1,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    for payload, period_sec in iter_script_payloads(plan, loops=loops):
        emit(payload)
        sleep(period_sec)


def _load_step(index: int, raw: Any) -> AudioScriptStep:
    if not isinstance(raw, Mapping):
        raise ValueError(f"audio script step {index} must be a JSON object")
    explicit = raw.get("message", raw.get("payload"))
    if explicit is not None and not isinstance(explicit, Mapping):
        raise ValueError(f"audio script step {index} message/payload must be a JSON object")
    payload = dict(explicit) if explicit is not None else {key: value for key, value in raw.items() if key not in _CONTROL_KEYS}
    if not payload:
        raise ValueError(f"audio script step {index} must define message fields")
    period_sec = _period(index, raw)
    count = int(raw.get("count", 1))
    if count <= 0:
        raise ValueError(f"audio script step {index} count must be greater than zero")
    return AudioScriptStep(payload=payload, period_sec=period_sec, count=count)


def _period(index: int, raw: Mapping[str, Any]) -> float:
    if raw.get("period_sec") is not None:
        period_sec = float(raw["period_sec"])
    elif raw.get("rate_hz") is not None:
        rate_hz = float(raw["rate_hz"])
        if rate_hz <= 0.0:
            raise ValueError(f"audio script step {index} rate_hz must be greater than zero")
        period_sec = 1.0 / rate_hz
    else:
        period_sec = 1.0
    if period_sec <= 0.0:
        raise ValueError(f"audio script step {index} period_sec must be greater than zero")
    return period_sec


def _clean_string(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("audio script contains an empty string field")
    return text
