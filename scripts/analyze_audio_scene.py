#!/usr/bin/env python3
"""Analyze a WAV/FLAC file using exactly the live music model and tempo code."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_detection.audio_scene import AudioSetClassifier, estimate_tempo


def main():
    import librosa
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    audio, rate = librosa.load(args.audio, sr=16000, mono=True, duration=args.seconds)
    result = AudioSetClassifier().classify(audio, rate)
    result.update(estimate_tempo(audio, rate))
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
