#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH_WHEEL="/home/qimao/grad_ws/package/torch-1.13.1-cp38-cp38-manylinux1_x86_64.whl"

python3 -m venv "${ROOT_DIR}/.venv"
source "${ROOT_DIR}/.venv/bin/activate"

python -m pip install --upgrade pip

if [ -f "${TORCH_WHEEL}" ]; then
  python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "${TORCH_WHEEL}"
else
  echo "Local torch wheel not found: ${TORCH_WHEEL}"
  echo "Please prepare a compatible torch wheel first."
  exit 1
fi

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r "${ROOT_DIR}/requirements.txt"

echo
echo "CPU inference environment is ready."
echo "Activate it with:"
echo "  source \"${ROOT_DIR}/.venv/bin/activate\""
