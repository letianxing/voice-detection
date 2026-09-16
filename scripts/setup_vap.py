#!/usr/bin/env python3
"""Fetch the pinned official VAP inference weights; no environment upgrades."""
from pathlib import Path
import hashlib,urllib.request
root=Path(__file__).resolve().parents[1]
target=root/'weights/vap/model.pt'
expected='043f2800bb95a7c27ad4271b1034fe2c1b70766f20e3b4ff91fe76ee9dc2360e'
url='https://raw.githubusercontent.com/ErikEkstedt/VoiceActivityProjection/f39a78b23a6dccdbedd106e00b48c410b8739f5d/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt'
if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.download')
    urllib.request.urlretrieve(url,temporary)
    if hashlib.sha256(temporary.read_bytes()).hexdigest()!=expected:raise RuntimeError('VAP model checksum mismatch')
    temporary.replace(target)
print('VAP weights verified:',target)
