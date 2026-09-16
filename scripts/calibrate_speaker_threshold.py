#!/usr/bin/env python3
"""Measure the speaker thresholds on the actual people who will use this.

The field session showed an enrolled speaker scoring 0.50 to 0.59 against a 0.60
reliability bar, so every path that depended on the voiceprint failed. My earlier
check used synthesised voices, which are far more self-consistent than people —
that was not enough to justify carrying thresholds onto a different model, and it
is why they were wrong.

This reads real recordings instead. Point it at a folder per person:

    recordings/
      小天/  a.wav b.wav c.wav
      小林/  a.wav b.wav

Each file: 16 kHz mono WAV, one person, two seconds or more, several different
sentences. Three or more people gives a usable impostor distribution; with one
person it can only report same-speaker scores and will say so.

    .venv/bin/python scripts/calibrate_speaker_threshold.py recordings/
"""
import argparse
import itertools
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "weights/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"


def read_wav(path):
    with wave.open(str(path)) as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"{path.name}: need 16-bit mono")
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, rate


def embed(extractor, audio, rate):
    stream = extractor.create_stream()
    stream.accept_waveform(rate, audio)
    stream.input_finished()
    return np.array(extractor.compute(stream), dtype=np.float64)


def cosine(left, right):
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="one subfolder per person, WAV files inside")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--min-seconds", type=float, default=1.5)
    args = parser.parse_args()

    import sherpa_onnx
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=args.model, num_threads=2, provider="cpu"))

    people = {}
    for folder in sorted(Path(args.folder).iterdir()):
        if not folder.is_dir():
            continue
        vectors = []
        for wav in sorted(folder.glob("*.wav")):
            audio, rate = read_wav(wav)
            if len(audio) / rate < args.min_seconds:
                print(f"  跳过 {wav.name}：只有 {len(audio)/rate:.1f} 秒")
                continue
            vectors.append(embed(extractor, audio, rate))
        if len(vectors) >= 2:
            people[folder.name] = vectors
        elif vectors:
            print(f"  {folder.name}: 只有 1 段可用，无法算同人分布")

    if not people:
        print("没有可用录音。每人至少 2 段、每段 ≥1.5 秒的 16kHz 单声道 WAV。")
        return 1

    same = [cosine(a, b) for vectors in people.values() for a, b in itertools.combinations(vectors, 2)]
    different = [cosine(a, b) for left, right in itertools.combinations(people, 2)
                 for a in people[left] for b in people[right]]

    print(f"\n模型 {Path(args.model).name}")
    print(f"{len(people)} 人：" + "、".join(f"{name}({len(v)}段)" for name, v in people.items()))
    print(f"\n同一人  n={len(same):3d}  最低 {min(same):.3f}  平均 {np.mean(same):.3f}  最高 {max(same):.3f}")
    if not different:
        print("\n只有一个人，算不出冒充者分布，无法定阈值。至少再录一个人。")
        return 1
    print(f"不同人  n={len(different):3d}  最低 {min(different):.3f}  平均 {np.mean(different):.3f}  最高 {max(different):.3f}")

    margin = min(same) - max(different)
    print(f"\n分离间隔（同人最低 − 不同人最高）= {margin:+.3f}")
    if margin <= 0:
        print("两个分布重叠：没有一个阈值能同时不漏不错。先改善录音质量或补更多样本，")
        print("再考虑降低对声纹的依赖，而不是挑一个中间值凑合。")

    candidates = np.unique(np.concatenate([same, different]))
    scored = []
    for threshold in candidates:
        false_reject = float(np.mean(np.asarray(same) < threshold))
        false_accept = float(np.mean(np.asarray(different) >= threshold))
        scored.append((abs(false_reject - false_accept) + false_reject + false_accept,
                       threshold, false_reject, false_accept))
    _, crossing, reject_at, accept_at = min(scored)
    print(f"等错误点 ≈ {crossing:.3f}（此处漏认 {reject_at:.0%}、错认 {accept_at:.0%}）")

    match = max(0.05, float(np.percentile(different, 99)) + 0.02)
    reliable = max(match + 0.05, float(np.percentile(same, 10)))
    print(f"\n建议：匹配阈值 {match:.2f}，可靠阈值 {reliable:.2f}")
    print("  匹配 = 冒充者分数的 99 分位再抬一点；可靠 = 同人分数的 10 分位。")
    print("  在 voice-detection 的声纹匹配处使用匹配阈值；")
    print("  注意力里任何要求「可靠声纹」的路径使用可靠阈值。")
    print("\n这是这几个人、这些录音、这个麦克风下的结果，不是通用准确率。换人换环境要重测。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
