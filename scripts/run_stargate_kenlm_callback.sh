#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-kenlm-callback"

KENLM_ROOT_INPUT="${KENLM_ROOT:-}"
COMPARE_MODE=0
ARGS=()

normalize_compare_language() {
    case "$1" in
        xh|zu)
            echo "auto"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kenlm-root)
            KENLM_ROOT_INPUT="$2"
            shift 2
            ;;
                --compare)
                        COMPARE_MODE=1
                        shift
                        ;;
        -h|--help)
            cat <<'EOF'
usage: scripts/run_stargate_kenlm_callback.sh [--compare] --kenlm-root /path/to/kenlm --model MODEL --audio AUDIO --kenlm-model LM [args...]

This wrapper configures and builds the standalone whisper-kenlm-callback example,
then runs it with the remaining arguments.

Required:
  --kenlm-root PATH   KenLM source or install root containing lm/model.hh and built libkenlm/libkenlm_util

Optional:
    --compare           print a baseline whisper-cli transcription before the rescored output

All remaining arguments are passed through to whisper-kenlm-callback.
EOF
            exit 0
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${KENLM_ROOT_INPUT}" ]]; then
    for candidate in "${ROOT_DIR}/build/kenlm" "${ROOT_DIR}/third_party/kenlm"; do
        if [[ -d "${candidate}" ]]; then
            KENLM_ROOT_INPUT="${candidate}"
            break
        fi
    done
fi

if [[ -z "${KENLM_ROOT_INPUT}" ]]; then
    echo "error: KenLM is not available on this host. Pass --kenlm-root /path/to/kenlm (source or install tree)." >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DKENLM_ROOT="${KENLM_ROOT_INPUT}" >/dev/null
cmake --build "${BUILD_DIR}" --target whisper-kenlm-callback -j4 >/dev/null

if [[ "${COMPARE_MODE}" == "1" ]]; then
    model_path=""
    audio_path=""
    language=""

    for ((i=0; i<${#ARGS[@]}; i++)); do
        case "${ARGS[$i]}" in
            --model)
                model_path="${ARGS[$((i+1))]:-}"
                ;;
            --audio)
                audio_path="${ARGS[$((i+1))]:-}"
                ;;
            --language)
                language="${ARGS[$((i+1))]:-}"
                ;;
        esac
    done

    if [[ -z "${model_path}" || -z "${audio_path}" ]]; then
        echo "error: --compare requires --model and --audio to be passed through to the harness" >&2
        exit 1
    fi

    echo "===== baseline whisper-cli ====="
    baseline_cmd=("${ROOT_DIR}/build/bin/whisper-cli" -m "${model_path}" -f "${audio_path}" -ng -nt)
    if [[ -n "${language}" ]]; then
        baseline_lang="$(normalize_compare_language "${language}")"
        if [[ "${baseline_lang}" != "${language}" ]]; then
            echo "note: whisper-cli does not accept '${language}' in this build; baseline fallback uses auto-detect"
        fi
        baseline_cmd+=(-l "${baseline_lang}")
    fi
    "${baseline_cmd[@]}"
    echo
    echo "===== kenlm callback ====="
fi

exec "${BUILD_DIR}/bin/whisper-kenlm-callback" "${ARGS[@]}"