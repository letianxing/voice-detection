#!/usr/bin/env python3
"""Download pinned, safetensors-only AudioSet AST weights for offline inference."""
from pathlib import Path
import hashlib
import json
from huggingface_hub import snapshot_download

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
root = Path(__file__).resolve().parents[1] / "weights" / "ast-audioset"
snapshot_download(MODEL, revision=REVISION, local_dir=root,
                  allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
manifest = {"model": MODEL, "revision": REVISION, "files": {}}
for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
    digest = hashlib.sha256()
    with (root / name).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest["files"][name] = digest.hexdigest()
(root / "provenance.json").write_text(json.dumps(manifest, indent=2))
print("Audio classifier ready:", root)

# Dedicated supervised ten-genre model; AudioSet tags alone are often too weak for genre decisions.
genre_root = root.parent / "music-genre"
genre_model = "dima806/music_genres_classification"
genre_revision = "5f71fb1e2c6bedcddb2bfb1e929fc70655780902"
snapshot_download(genre_model, revision=genre_revision, local_dir=genre_root,
                  allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
manifest = {"model": genre_model, "revision": genre_revision, "files": {}}
for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
    digest = hashlib.sha256()
    with (genre_root / name).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest["files"][name] = digest.hexdigest()
(genre_root / "provenance.json").write_text(json.dumps(manifest, indent=2))
print("Genre classifier ready:", genre_root)
