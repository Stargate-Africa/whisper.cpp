#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WHISPER_CLI = REPO_ROOT / "build" / "bin" / "whisper-cli"
KENLM_WRAPPER = REPO_ROOT / "scripts" / "run_stargate_kenlm_callback.sh"


def normalize_language_for_cli(language: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized in {"xh", "zu"}:
        return "auto"
    return normalized or "auto"


def normalize_text(text: str) -> list[str]:
    normalized = re.sub(r"[^\w'\-]+", " ", (text or "").lower())
    return [token for token in normalized.split() if token]


def wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference)
    hyp_words = normalize_text(hypothesis)

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dp = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        dp[row][0] = row
    for col in range(cols):
        dp[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            substitution = 0 if ref_words[row - 1] == hyp_words[col - 1] else 1
            dp[row][col] = min(
                dp[row - 1][col] + 1,
                dp[row][col - 1] + 1,
                dp[row - 1][col - 1] + substitution,
            )

    return dp[-1][-1] / len(ref_words)


def run_command(command: list[str], *, cwd: Path) -> tuple[str, str, float]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"command failed: {' '.join(shlex.quote(part) for part in command)}\n{detail}")
    return completed.stdout.strip(), completed.stderr.strip(), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline whisper.cpp vs KenLM callback WER on a provided reference transcript.")
    parser.add_argument("--model", required=True, help="Whisper GGML model path, relative to repo root or absolute")
    parser.add_argument("--audio", required=True, help="Audio path, relative to repo root or absolute")
    parser.add_argument("--language", required=True, help="Requested language code/name")
    parser.add_argument("--kenlm-model", required=True, help="KenLM binary path, relative to repo root or absolute")
    parser.add_argument("--kenlm-root", required=True, help="KenLM source or install root used by the standalone callback wrapper")
    parser.add_argument("--reference-file", help="Path to a UTF-8 reference transcript file")
    parser.add_argument("--reference-text", help="Reference transcript text")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--rescore-top-k", type=int, default=1)
    parser.add_argument("--lm-alpha", type=float, default=0.25)
    parser.add_argument("--hotword-bias", type=float, default=0.0)
    parser.add_argument("--hotword", action="append", default=[], help="Optional hotword for the callback harness")
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (REPO_ROOT / path).resolve()


def main() -> int:
    args = parse_args()

    if not args.reference_file and not args.reference_text:
        print("error: provide --reference-file or --reference-text", file=sys.stderr)
        return 2

    if args.reference_file and args.reference_text:
        print("error: provide only one of --reference-file or --reference-text", file=sys.stderr)
        return 2

    model_path = resolve_path(args.model)
    audio_path = resolve_path(args.audio)
    kenlm_model_path = resolve_path(args.kenlm_model)
    kenlm_root = resolve_path(args.kenlm_root)

    if args.reference_file:
        reference_text = resolve_path(args.reference_file).read_text(encoding="utf-8").strip()
    else:
        reference_text = args.reference_text.strip()

    baseline_language = normalize_language_for_cli(args.language)
    baseline_cmd = [
        str(WHISPER_CLI),
        "-m", str(model_path),
        "-f", str(audio_path),
        "-l", baseline_language,
        "-ng",
        "-nt",
    ]

    rescored_cmd = [
        str(KENLM_WRAPPER),
        "--kenlm-root", str(kenlm_root),
        "--model", str(model_path),
        "--audio", str(audio_path),
        "--kenlm-model", str(kenlm_model_path),
        "--language", args.language,
        "--beam-size", str(args.beam_size),
        "--rescore-top-k", str(args.rescore_top_k),
        "--lm-alpha", str(args.lm_alpha),
        "--hotword-bias", str(args.hotword_bias),
    ]
    for hotword in args.hotword:
        rescored_cmd.extend(["--hotword", hotword])

    baseline_out, baseline_err, baseline_time = run_command(baseline_cmd, cwd=REPO_ROOT)
    rescored_out, rescored_err, rescored_time = run_command(rescored_cmd, cwd=REPO_ROOT)

    baseline_wer = wer(reference_text, baseline_out)
    rescored_wer = wer(reference_text, rescored_out)

    print("===== reference =====")
    print(reference_text)
    print()
    print("===== baseline =====")
    print(baseline_out)
    print(f"time_s={baseline_time:.2f} wer={baseline_wer:.4f}")
    if baseline_err:
        print(f"stderr_tail={baseline_err.splitlines()[-1]}")
    print()
    print("===== rescored =====")
    print(rescored_out)
    print(f"time_s={rescored_time:.2f} wer={rescored_wer:.4f}")
    if rescored_err:
        print(f"stderr_tail={rescored_err.splitlines()[-1]}")
    print()
    print("===== delta =====")
    print(f"wer_delta={baseline_wer - rescored_wer:+.4f}")
    print(f"time_delta_s={rescored_time - baseline_time:+.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())