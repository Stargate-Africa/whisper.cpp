#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="${ROOT_DIR}/bindings/python"
DEFAULT_KENLM_DIR="$(cd "${ROOT_DIR}/../.." && pwd)/third_party/kenlm"
LEGACY_KENLM_DIR="${ROOT_DIR}/build/kenlm"

resolve_kenlm_root() {
    if [[ -n "${KENLM_ROOT:-}" ]]; then
        printf '%s\n' "${KENLM_ROOT}"
        return
    fi

    for candidate in \
        "${DEFAULT_KENLM_DIR}" \
        "${LEGACY_KENLM_DIR}"; do
        if [[ -d "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done

    printf '%s\n' "${DEFAULT_KENLM_DIR}"
}

export KENLM_ROOT="$(resolve_kenlm_root)"

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
