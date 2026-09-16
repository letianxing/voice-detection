#!/usr/bin/env python3
"""Measure the cosine score distribution of a speaker model before trusting it.

Swapping the embedding model invalidates the old thresholds: scores from a
different network are not on the same scale. This renders a set of distinct
macOS voices, computes same-voice and different-voice cosine distributions, and
prints where a separating threshold would sit.

These are synthesised voices, not people. The numbers bound the impostor floor
and show that a threshold carried over from another model would be wrong; they
are not a field EER and must never be quoted as identification accuracy. No
microphone or camera is opened.

    .venv/bin/python scripts/measure_speaker_thresholds.py
"""
import argparse
import itertools
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VOICES = ["Tingting", "Meijia", "Eddy (中文（中国大陆）)", "Flo (中文（中国大陆）)",
          "Grandpa (中文（中国大陆）)", "Rocko (中文（中国大陆）)"]
SENTENCES = ["今天天气不错，我们下午去公园散散步吧",
             "这个问题我还没想清楚，需要再看一会儿资料",
             "周六上午有空的话，一起去看看新开的那家店",
             "刚才你说的那件事，我觉得可以再商量一下"]


def render(voice, text, path):
    subprocess.run(["/usr/bin/say", "-v", voice, "-o", str(path), "--file-format=WAVE",
                    "--data-format=LEI16@16000", text], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with wave.open(str(path)) as handle:
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def embed(extractor, audio):
    stream = extractor.create_stream()
    stream.accept_waveform(16000, audio)
    stream.input_finished()
    return np.array(extractor.compute(stream), dtype=np.float64)


def cosine(left, right):
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


def measure(model_path, voices, sentences):
    import sherpa_onnx
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=2, provider="cpu")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    embeddings = {}
    with tempfile.TemporaryDirectory(prefix="speaker-threshold-") as directory:
        path = Path(directory) / "sample.wav"
        for voice in voices:
            samples = []
            for text in sentences:
                try:
                    samples.append(embed(extractor, render(voice, text, path)))
                except subprocess.CalledProcessError:
                    break
            if len(samples) >= 2:
                embeddings[voice] = samples

    same, different = [], []
    for voice, samples in embeddings.items():
        same.extend(cosine(a, b) for a, b in itertools.combinations(samples, 2))
    for left, right in itertools.combinations(embeddings, 2):
        different.extend(cosine(a, b) for a in embeddings[left] for b in embeddings[right])
    return embeddings, np.array(same), np.array(different)


def report(label, same, different):
    if same.size == 0 or different.size == 0:
        print(f"{label}: not enough voices rendered to measure anything")
        return None
    # Equal-error style crossing point over the observed samples only.
    candidates = np.unique(np.concatenate([same, different]))
    best, best_cost = None, None
    for threshold in candidates:
        false_reject = float((same < threshold).mean())
        false_accept = float((different >= threshold).mean())
        cost = abs(false_reject - false_accept) + false_reject + false_accept
        if best_cost is None or cost < best_cost:
            best, best_cost = threshold, cost
    margin = float(same.min() - different.max())
    print(f"{label}")
    print(f"  same voice      n={same.size:3d}  min={same.min():.3f} mean={same.mean():.3f} max={same.max():.3f}")
    print(f"  different voice n={different.size:3d}  min={different.min():.3f} mean={different.mean():.3f} max={different.max():.3f}")
    print(f"  separation margin (same.min - different.max) = {margin:+.3f}")
    print(f"  crossing point on these samples ≈ {best:.3f}")
    print(f"  suggested match / reliable = {max(0.0, best - 0.06):.2f} / {min(0.95, best + 0.10):.2f}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=[
        "weights/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "weights/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"])
    args = parser.parse_args()

    print("合成语音测量，不是真人；只用于确认阈值不能跨模型平移，不能当作现场识别准确率。\n")
    for model in args.models:
        path = ROOT / model if not Path(model).is_absolute() else Path(model)
        if not path.exists():
            print(f"{path.name}: 缺少模型文件，跳过\n")
            continue
        voices, same, different = measure(path, VOICES, SENTENCES)
        print(f"{path.name}  ({len(voices)} 个可用音色 × {len(SENTENCES)} 句)")
        report("  ", same, different)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
