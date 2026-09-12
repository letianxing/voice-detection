from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass

from .frontend import now_ms


@dataclass(frozen=True)
class PlaybackState:
    stamp_ms: int
    speaking: bool
    text: str
    backend: str
    started_ms: int | None = None
    ended_ms: int | None = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, separators=(",", ":"))


def speak_text(text: str, backend: str = "auto", dry_run: bool = False) -> PlaybackState:
    selected = choose_backend(backend)
    started_ms = now_ms()
    if dry_run:
        return PlaybackState(
            stamp_ms=started_ms,
            speaking=False,
            text=text,
            backend=f"{selected}:dry_run",
            started_ms=started_ms,
            ended_ms=started_ms,
        )
    if selected == "macos_say":
        subprocess.run(["say", text], check=True)
    elif selected == "stdout":
        print(text, flush=True)
    else:
        raise ValueError(f"unsupported TTS backend: {selected}")
    ended_ms = now_ms()
    return PlaybackState(
        stamp_ms=ended_ms,
        speaking=False,
        text=text,
        backend=selected,
        started_ms=started_ms,
        ended_ms=ended_ms,
    )


def choose_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    if platform.system() == "Darwin" and shutil.which("say"):
        return "macos_say"
    return "stdout"
