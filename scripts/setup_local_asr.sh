#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${ROOT_DIR}/weights"
VOSK_NAME="vosk-model-small-cn-0.22"
VOSK_DIR="${CACHE_DIR}/${VOSK_NAME}"
VOSK_ZIP="${CACHE_DIR}/${VOSK_NAME}.zip"
VOSK_URL="https://alphacephei.com/vosk/models/${VOSK_NAME}.zip"
PYTHON_BIN="${PYTHON_BIN:-/Users/letianxing/Golands/robot-attention-perception/.venv-mac/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "run robot-attention-perception/scripts/setup_mac_first_test.sh first" >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install vosk
mkdir -p "${CACHE_DIR}"
if [[ ! -d "${VOSK_DIR}" ]]; then
  curl -L --fail --retry 5 --retry-delay 5 --retry-all-errors \
    --continue-at - --output "${VOSK_ZIP}" "${VOSK_URL}"
  ditto -x -k "${VOSK_ZIP}" "${CACHE_DIR}"
fi

if [[ ! -s "${VOSK_DIR}/conf/model.conf" ]]; then
  echo "Vosk model extraction failed: ${VOSK_DIR}" >&2
  exit 1
fi

echo "Vosk Chinese smoke-test model ready: ${VOSK_DIR}"

if [[ "${INSTALL_WHISPER_MODEL:-0}" == "1" ]]; then
  WHISPER_PATH="${CACHE_DIR}/ggml-base.bin"
  WHISPER_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
  EXPECTED_SHA1="465707469ff3a37a2b9b8d8f89f2f99de7299dac"
  command -v whisper-cli >/dev/null 2>&1 || brew install whisper-cpp
  curl -L --fail --retry 5 --retry-delay 5 --retry-all-errors \
    --continue-at - --output "${WHISPER_PATH}" "${WHISPER_URL}"
  ACTUAL_SHA1="$(shasum "${WHISPER_PATH}" | awk '{print $1}')"
  [[ "${ACTUAL_SHA1}" == "${EXPECTED_SHA1}" ]] || {
    echo "Whisper model checksum mismatch" >&2
    exit 1
  }
  echo "whisper.cpp model ready: ${WHISPER_PATH}"
fi
