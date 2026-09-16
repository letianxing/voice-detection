#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# weights/, because that is where everything in the code looks: the speaker
# embedder, REAL-TSE, VAP and the streaming recogniser all resolve under it.
# This script used to write to models/, so a fresh checkout downloaded files
# nothing ever read and the old models/ copy stayed behind.
MODEL_DIR="${ROOT_DIR}/weights"
mkdir -p "${MODEL_DIR}"

SPEAKER_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"
SPEAKER_PATH="${MODEL_DIR}/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"

if [[ ! -s "${SPEAKER_PATH}" ]]; then
  curl -L --fail --retry 5 --retry-delay 2 --retry-all-errors \
    --continue-at - --output "${SPEAKER_PATH}" "${SPEAKER_URL}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" -m pip install "sherpa-onnx>=1.13.8" >/dev/null
echo "speaker model ready: ${SPEAKER_PATH}"
echo "optional Chinese ASR: scripts/setup_local_asr.sh"

# Offline recogniser for the finished sentence. The streaming model keeps doing
# partials; this only re-reads completed utterances, where accuracy matters more
# than latency.
SENSEVOICE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
SENSEVOICE_DIR="${MODEL_DIR}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
if [ ! -f "${SENSEVOICE_DIR}/model.int8.onnx" ]; then
  curl -L --retry 3 -o "${MODEL_DIR}/sense-voice.tar.bz2" "${SENSEVOICE_URL}"
  tar xjf "${MODEL_DIR}/sense-voice.tar.bz2" -C "${MODEL_DIR}"
  rm -f "${MODEL_DIR}/sense-voice.tar.bz2"
fi
