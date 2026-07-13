#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt
if [ "${INSTALL_GPTOSS_KERNELS:-1}" = "1" ]; then
  "${VENV_DIR}/bin/python" -m pip install kernels
  "${VENV_DIR}/bin/python" -m pip install --no-deps "triton==3.4.0"
fi
"${VENV_DIR}/bin/python" -m pip install -e .
