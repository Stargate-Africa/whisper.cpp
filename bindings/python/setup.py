from __future__ import annotations

import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).resolve().parent
WHISPER_ROOT = ROOT.parent.parent
REPO_ROOT = WHISPER_ROOT.parent.parent
WHISPER_INCLUDE = WHISPER_ROOT / "include"
GGML_INCLUDE = WHISPER_ROOT / "ggml" / "include"
WHISPER_LIB_DIR = WHISPER_ROOT / "build" / "src"

def _resolve_kenlm_root() -> Path:
    override = os.getenv("KENLM_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    for candidate in (
        REPO_ROOT / "third_party" / "kenlm",
        WHISPER_ROOT / "build" / "kenlm",
        WHISPER_ROOT / "third_party" / "kenlm",
    ):
        if candidate.exists():
            return candidate.resolve()

    return (REPO_ROOT / "third_party" / "kenlm").resolve()


KENLM_ROOT = _resolve_kenlm_root()
KENLM_LIB_DIR = KENLM_ROOT / "build" / "lib"
KENLM_LIB = KENLM_LIB_DIR / "libkenlm.a"
KENLM_UTIL_LIB = KENLM_LIB_DIR / "libkenlm_util.a"

if not KENLM_LIB.is_file() or not KENLM_UTIL_LIB.is_file():
    raise RuntimeError(
        "KenLM static libraries were not found. Set KENLM_ROOT or build KenLM under "
        f"{KENLM_ROOT}. Expected {KENLM_LIB} and {KENLM_UTIL_LIB}."
    )

ext_modules = [
    Pybind11Extension(
        "stargate_whispercpp_binding._whispercpp_binding",
        [str(ROOT / "stargate_whispercpp_binding.cpp")],
        include_dirs=[str(WHISPER_INCLUDE), str(GGML_INCLUDE), str(KENLM_ROOT)],
        library_dirs=[str(WHISPER_LIB_DIR)],
        libraries=["whisper", "z", "bz2", "lzma", "pthread"],
        extra_objects=[str(KENLM_LIB), str(KENLM_UTIL_LIB)],
        define_macros=[("KENLM_MAX_ORDER", "6")],
        cxx_std=17,
        extra_link_args=[f"-Wl,-rpath,{WHISPER_LIB_DIR}"],
    )
]

setup(
    name="stargate-whispercpp-binding",
    version="0.0.1",
    packages=["stargate_whispercpp_binding"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

