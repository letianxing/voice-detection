Source: https://github.com/ErikEkstedt/VoiceActivityProjection
Commit: f39a78b23a6dccdbedd106e00b48c410b8739f5d
License: MIT (see LICENSE)
Model SHA256: 043f2800bb95a7c27ad4271b1034fe2c1b70766f20e3b4ff91fe76ee9dc2360e
Local change: load static CPC architecture config when load_pretrained=0, avoiding an unnecessary network download; the supplied full VAP state dict contains encoder weights. Inference calls forward and probability aggregation, avoiding upstream probs() training-loss calculation.
