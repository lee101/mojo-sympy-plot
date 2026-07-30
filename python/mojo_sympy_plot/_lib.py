"""ctypes access to the Mojo expression evaluator."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_SYMPY_PLOT_LIB") or os.path.join(
    ROOT, "dist", "libmojo-sympy-plot.so"
)
I = ctypes.c_int64

_library: ctypes.CDLL | None = None


class BuildError(RuntimeError):
    pass


def build() -> str:
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            build()
        _library = ctypes.CDLL(LIB)
        _library.msp_evaluate.argtypes = [I] * 13
        _library.msp_evaluate.restype = I
    return _library


def addr(value: np.ndarray) -> int:
    address = value.ctypes.data
    if not address:
        raise ValueError("cannot pass an empty buffer across the Mojo FFI")
    return address
