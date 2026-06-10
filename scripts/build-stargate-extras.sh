#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
CALLBACK_BUILD_DIR="${ROOT_DIR}/build-kenlm-callback"
KENLM_REPO_URL="${KENLM_REPO_URL:-https://github.com/kpu/kenlm.git}"

BACKEND="cpu"
BUILD_CORE=1
BUILD_BINDING=1
BUILD_CALLBACK=1
BUILD_KENLM_FROM_SOURCE="${BUILD_KENLM_FROM_SOURCE:-1}"
INSTALL_PYTHON=0
FORCE_CLEAN=0
KENLM_ROOT_INPUT="${KENLM_ROOT:-${ROOT_DIR}/build/kenlm}"

usage() {
    cat <<'EOF'
Build Stargate-specific whisper.cpp extras from inside this whisper.cpp checkout.

Usage:
  scripts/build-stargate-extras.sh [options]

Options:
  --backend {cpu|cuda|hip}   Backend used to build libwhisper and the callback.
                             Default: cpu
  --kenlm-root DIR           Existing KenLM source/build root.
                             Default: build/kenlm
  --no-core                  Skip rebuilding libwhisper.
  --no-binding               Skip building the Stargate Python binding.
  --no-callback              Skip building whisper-kenlm-callback.
  --install-python           Run 'python -m pip install -e bindings/python' after build.
  --clean                    Remove build directories before rebuilding.
  -h, --help                 Show this help text.

Environment:
  KENLM_ROOT                 Same as --kenlm-root.
  BUILD_KENLM_FROM_SOURCE    1 to clone/build KenLM if missing (default: 1).
  KENLM_REPO_URL             KenLM git remote used when cloning.
  CMAKE_ARGS                 Extra args appended to all cmake configure invocations.
  CUDA_ARCHITECTURES         Passed through for --backend cuda.
  AMDGPU_TARGETS             Passed through for --backend hip.

Examples:
  scripts/build-stargate-extras.sh --backend cpu
  scripts/build-stargate-extras.sh --backend cuda --kenlm-root /opt/kenlm
  scripts/build-stargate-extras.sh --backend hip --no-binding
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

sync_git_checkout() {
    local target_dir="$1"
    local repo_url="$2"
    local branch="$3"
    local label="$4"

    if [[ -d "${target_dir}/.git" ]]; then
        if [[ -n "$(git -C "${target_dir}" status --porcelain --untracked-files=no)" ]]; then
            die "${label} checkout has local tracked changes: ${target_dir}"
        fi
        echo "Updating ${label}: ${repo_url} (${branch})"
        git -C "${target_dir}" remote set-url origin "${repo_url}"
        git -C "${target_dir}" fetch --depth 1 origin "${branch}"
        git -C "${target_dir}" checkout -B "${branch}" FETCH_HEAD
    else
        echo "Cloning ${label}: ${repo_url} (${branch})"
        mkdir -p "$(dirname "${target_dir}")"
        git clone --depth 1 --branch "${branch}" "${repo_url}" "${target_dir}"
    fi
}

ensure_kenlm() {
    if [[ -f "${KENLM_ROOT_INPUT}/build/lib/libkenlm.a" && -f "${KENLM_ROOT_INPUT}/build/lib/libkenlm_util.a" ]]; then
        echo "KenLM build found: ${KENLM_ROOT_INPUT}"
        return
    fi

    if [[ "${BUILD_KENLM_FROM_SOURCE}" != "1" ]]; then
        die "KenLM static libraries not found under ${KENLM_ROOT_INPUT}/build/lib. Set KENLM_ROOT, pass --kenlm-root, or enable BUILD_KENLM_FROM_SOURCE=1."
    fi

    require_cmd git
    sync_git_checkout "${KENLM_ROOT_INPUT}" "${KENLM_REPO_URL}" "master" "KenLM"
    echo "Building KenLM: ${KENLM_ROOT_INPUT}"
    cmake -S "${KENLM_ROOT_INPUT}" -B "${KENLM_ROOT_INPUT}/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${KENLM_ROOT_INPUT}/build" --config Release -j "$(nproc)"
}

append_cmake_args() {
    local -n out_ref=$1

    case "${BACKEND}" in
        cpu)
            ;;
        cuda)
            require_cmd nvcc
            out_ref+=("-DGGML_CUDA=ON")
            if [[ -n "${CUDA_ARCHITECTURES:-}" ]]; then
                out_ref+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}")
            fi
            ;;
        hip)
            require_cmd hipcc
            out_ref+=("-DGGML_HIP=ON")
            if [[ -n "${AMDGPU_TARGETS:-}" ]]; then
                out_ref+=("-DAMDGPU_TARGETS=${AMDGPU_TARGETS}")
            fi
            ;;
        *)
            die "Unsupported backend: ${BACKEND}"
            ;;
    esac

    if [[ -n "${CMAKE_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        local extra_args=( ${CMAKE_ARGS} )
        out_ref+=("${extra_args[@]}")
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            BACKEND="$2"
            shift 2
            ;;
        --kenlm-root)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            KENLM_ROOT_INPUT="$2"
            shift 2
            ;;
        --no-core)
            BUILD_CORE=0
            shift
            ;;
        --no-binding)
            BUILD_BINDING=0
            shift
            ;;
        --no-callback)
            BUILD_CALLBACK=0
            shift
            ;;
        --install-python)
            INSTALL_PYTHON=1
            shift
            ;;
        --clean)
            FORCE_CLEAN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

require_cmd cmake
require_cmd python3

KENLM_ROOT_INPUT="$(python3 - <<'PY' "${KENLM_ROOT_INPUT}"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

if [[ ${BUILD_BINDING} -eq 1 || ${BUILD_CALLBACK} -eq 1 ]]; then
    ensure_kenlm
fi

if [[ ${FORCE_CLEAN} -eq 1 ]]; then
    rm -rf "${BUILD_DIR}" "${CALLBACK_BUILD_DIR}" "${ROOT_DIR}/bindings/python/build"
    find "${ROOT_DIR}/bindings/python/stargate_whispercpp_binding" -name '_whispercpp_binding*.so' -delete
fi

declare -a core_cmake_args=("-DCMAKE_BUILD_TYPE=Release")
append_cmake_args core_cmake_args

if [[ ${BUILD_CORE} -eq 1 ]]; then
    echo "Configuring whisper.cpp core build (${BACKEND})"
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" "${core_cmake_args[@]}"
    echo "Building libwhisper (${BACKEND})"
    cmake --build "${BUILD_DIR}" --config Release -j "$(nproc)"
fi

if [[ ${BUILD_BINDING} -eq 1 ]]; then
    if [[ ! -x "${ROOT_DIR}/bindings/python/build_stargate_whispercpp_binding.sh" ]]; then
        die "Stargate binding build script missing: ${ROOT_DIR}/bindings/python/build_stargate_whispercpp_binding.sh"
    fi
    export KENLM_ROOT="${KENLM_ROOT_INPUT}"
    echo "Building Stargate Python binding"
    "${ROOT_DIR}/bindings/python/build_stargate_whispercpp_binding.sh"
    if [[ ${INSTALL_PYTHON} -eq 1 ]]; then
        echo "Installing binding into current Python environment"
        python3 -m pip install -e "${ROOT_DIR}/bindings/python"
    fi
fi

if [[ ${BUILD_CALLBACK} -eq 1 ]]; then
    if [[ ! -d "${ROOT_DIR}/examples/kenlm-callback" ]]; then
        die "KenLM callback example missing: ${ROOT_DIR}/examples/kenlm-callback"
    fi
    declare -a callback_cmake_args=("-DCMAKE_BUILD_TYPE=Release" "-DKENLM_ROOT=${KENLM_ROOT_INPUT}")
    append_cmake_args callback_cmake_args
    echo "Configuring whisper-kenlm-callback (${BACKEND})"
    cmake -S "${ROOT_DIR}" -B "${CALLBACK_BUILD_DIR}" "${callback_cmake_args[@]}"
    echo "Building whisper-kenlm-callback"
    cmake --build "${CALLBACK_BUILD_DIR}" --target whisper-kenlm-callback -j "$(nproc)"
fi

echo
echo "Done."
echo "  backend=${BACKEND}"
echo "  whisper_build=${BUILD_DIR}"
if [[ ${BUILD_BINDING} -eq 1 ]]; then
    echo "  binding_package=${ROOT_DIR}/bindings/python/stargate_whispercpp_binding"
fi
if [[ ${BUILD_CALLBACK} -eq 1 ]]; then
    echo "  callback_binary=${CALLBACK_BUILD_DIR}/bin/whisper-kenlm-callback"
fi
echo "  kenlm_root=${KENLM_ROOT_INPUT}"