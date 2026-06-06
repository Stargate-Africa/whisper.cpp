from __future__ import annotations

import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).resolve().parent
WHISPER_ROOT = ROOT.parent.parent
WHISPER_INCLUDE = WHISPER_ROOT / "include"
GGML_INCLUDE = WHISPER_ROOT / "ggml" / "include"
WHISPER_LIB_DIR = WHISPER_ROOT / "build" / "src"

KENLM_ROOT = Path(os.getenv("KENLM_ROOT", "/home/ianfe/src/KenLM/third_party/kenlm")).resolve()
KENLM_LIB_DIR = KENLM_ROOT / "build" / "lib"
KENLM_LIB = KENLM_LIB_DIR / "libkenlm.a"
KENLM_UTIL_LIB = KENLM_LIB_DIR / "libkenlm_util.a"

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
