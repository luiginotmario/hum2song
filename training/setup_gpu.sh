#!/usr/bin/env bash
# Install the Phase 1 training stack on an Ubuntu GPU box (Lambda A100 or similar).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ffmpeg libsndfile1
fi
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r training/requirements.txt
python -m pip install -e .
echo "venv ready. Activate with: source ${ROOT}/.venv/bin/activate"
