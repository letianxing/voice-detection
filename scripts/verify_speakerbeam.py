#!/usr/bin/env python3
"""Load the REAL-TSE causal SpeakerBeam checkpoint and run a CPU smoke test."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torchaudio

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *args, **kwargs: None
sox = types.ModuleType("torchaudio.sox_effects")
sox.apply_effects_tensor = lambda waveform, sample_rate, effects: (waveform, sample_rate)
sys.modules.setdefault("torchaudio.sox_effects", sox)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "real_tse"))

from wesep.cli.extractor import Extractor  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model = Extractor(str(root / "weights" / "real-tse" / "pretrained" / "spk_emb_causal_100"))
    mix = torch.zeros(1, 16000)
    enroll = torch.zeros(1, 16000)
    output = model.extract_speech_from_pcm(mix, 16000, enroll, 16000)
    print({"model": "spk_emb_causal_100", "causal": True, "output_shape": tuple(output.shape)})


if __name__ == "__main__":
    main()
