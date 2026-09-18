#!/usr/bin/env python3
"""Teach the store the robot's own voice, so it can recognise itself.

Three defences already stand between the robot's playback and its memory: the
text it heard against the text it played, the echo residual left after
cancellation, and Brain dropping anything that is almost exactly what it just
said. All three are about the *sound* or the *words* of one utterance, and the
field found the gap between them: 「天行，我在这儿。」 came back as
「天晴我带着。」 — 0.40 text similarity, under every text bar — and was filed
under a stranger cluster, which had by then collected enough of the robot's
own voice to match it at 0.494.

A voiceprint of the synthesiser does not care what the recogniser made of the
words. It is the same voice every time, it is free to enrol, and it makes
"this is me" a matter of identity rather than of thresholds.

Synthesises a few sentences with the same macOS voice Brain speaks with, at
the same rate, and enrols them as the identity `robot`. Nothing is played: the
audio goes to a file. Run it again after changing BRAIN_TTS_VOICE.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_detection.speaker_embedding import SpeakerEmbedder, SpeakerProfileStore

# Ordinary replies rather than tongue twisters: the pool should look like what
# the microphone actually hears coming back.
LINES = ("天行，我在这儿。", "上海今天多云，挺舒服的。", "好呀，我记下了。",
         "嗯，我在听，你慢慢说。", "现在是下午两点半。")


def synthesise(text: str, voice: str, rate: int, folder: Path) -> tuple[np.ndarray, int]:
    """One line of speech as 16 kHz mono, written to disk, never played."""
    aiff = folder / "line.aiff"
    wav = folder / "line.wav"
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text],
                   check=True, capture_output=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    str(aiff), str(wav)], check=True, capture_output=True)
    with wave.open(str(wav)) as handle:
        frames = handle.readframes(handle.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, handle.getframerate()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default=os.environ.get("BRAIN_TTS_VOICE", "Tingting"))
    parser.add_argument("--rate", type=int, default=int(os.environ.get("BRAIN_TTS_RATE", "210")))
    parser.add_argument("--store", default=os.environ.get(
        "VOICE_SPEAKER_PROFILE_STORE", str(root / "config/speaker_profiles.json")))
    parser.add_argument("--model", default=os.environ.get(
        "VOICE_SPEAKER_MODEL",
        str(root / "weights/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx")))
    parser.add_argument("--identity", default="robot")
    parser.add_argument("--prune-matching", type=float, default=0.45,
                        help="auto-created clusters this similar to the robot's voice are dropped: "
                             "they are the robot, recorded before it knew itself")
    args = parser.parse_args()

    embedder = SpeakerEmbedder(args.model)
    if not embedder.ready:
        print("声纹模型加载失败：" + embedder.error, file=sys.stderr)
        return 1
    store = SpeakerProfileStore(args.store)

    vectors = []
    with tempfile.TemporaryDirectory() as folder:
        for line in LINES:
            audio, rate = synthesise(line, args.voice, args.rate, Path(folder))
            vector = embedder.embed(audio, rate)
            if vector is None:
                print(f"  跳过（太短或嵌入失败）：{line}")
                continue
            vectors.append(vector)
            print(f"  {line}  {len(audio)/rate:.2f}s")
    if len(vectors) < 2:
        print("合成样本不足，未写入", file=sys.stderr)
        return 1

    agreement = [float(np.dot(vectors[0], other)) for other in vectors[1:]]
    print(f"同一个合成音之间的一致度：{min(agreement):.3f} – {max(agreement):.3f}")
    store.enroll(args.identity, vectors[0], "robot", enrollment_text=LINES[0], anchor_floor=0.0)
    for vector in vectors[1:]:
        try:
            store.enroll(args.identity, vector, "robot", enrollment_text=LINES[0], anchor_floor=0.35)
        except ValueError as error:
            print(f"  一段没能进池：{error}")

    # Clusters that are actually the robot, minted before it had a voiceprint.
    dropped = []
    for vector in vectors:
        for row in store.rank(vector):
            if row["role"] == "stranger" and row["score"] >= args.prune_matching:
                dropped.append((row["speaker_id"], round(row["score"], 3)))
    for speaker_id, score in sorted(set(dropped)):
        print(f"  删掉自动聚类 {speaker_id}（与机器人自己的声音 {score}）")
        store.forget(speaker_id)

    profile = store.profiles[args.identity]
    print(f"已登记 {args.identity}：{len(profile['embeddings'])} 段，{args.voice} @ {args.rate}")
    # A running service holds its own copy and rewrites the whole file when it
    # saves, so it has to be told; otherwise this enrolment is invisible to it
    # until a restart.
    try:
        from urllib.request import build_opener, ProxyHandler, Request
        request = Request("http://127.0.0.1:8090/api/reload-profiles", data=b"{}", method="POST",
                          headers={"content-type": "application/json"})
        with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
            identities = json.loads(response.read().decode()).get("identities") or []
        print("运行中的 8090 已重新载入：" + "、".join(identities))
    except Exception:
        print("8090 没在跑（或拒绝连接）：下次启动时会读到")
    print("现在机器人听见自己说话时，声纹会直接报 robot，不再依赖文本相似度。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
