#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models"
mkdir -p "${MODEL_DIR}"

SPEAKER_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
SPEAKER_PATH="${MODEL_DIR}/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"

if [[ ! -s "${SPEAKER_PATH}" ]]; then
  curl -L --fail --retry 5 --retry-delay 2 --retry-all-errors \
    --continue-at - --output "${SPEAKER_PATH}" "${SPEAKER_URL}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" -m pip install "sherpa-onnx>=1.13.8" >/dev/null
echo "speaker model ready: ${SPEAKER_PATH}"
echo "optional Chinese ASR: scripts/setup_local_asr.sh"
