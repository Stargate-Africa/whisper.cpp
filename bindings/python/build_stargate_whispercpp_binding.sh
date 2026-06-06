#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="${ROOT_DIR}/bindings/python"

export KENLM_ROOT="${KENLM_ROOT:-/home/ianfe/src/KenLM/third_party/kenlm}"

if [[ ! -f "${ROOT_DIR}/build/src/libwhisper.so" ]]; then
    echo "error: libwhisper.so not found under ${ROOT_DIR}/build/src" >&2
    exit 1
fi

if [[ ! -f "${KENLM_ROOT}/build/lib/libkenlm.a" ]]; then
    echo "error: libkenlm.a not found under ${KENLM_ROOT}/build/lib" >&2
    exit 1
fi

cd "${PYTHON_DIR}"
python setup.py build_ext --inplace
