"""Benchmark Mojo sampling against SymPy's plotting series on identical grids."""

from __future__ import annotations

import gc
import math
import os
import platform
import sys
import time

import sympy as sp
from sympy.plotting import series as upstream

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_sympy_plot as msp  # noqa: E402

x, y, t = sp.symbols("x y t")


def timeit(function, repeat=2):
    best = math.inf
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def cases():
    yield (
        "identity line, 1M",
        msp.LineOver1DRangeSeries(
            x, (x, -10, 10), adaptive=False, n=1_000_000
        ).get_data,
        upstream.LineOver1DRangeSeries(
            x, (x, -10, 10), adaptive=False, n=1_000_000
        ).get_data,
    )

    yield (
        "default sine line, 1k",
        msp.LineOver1DRangeSeries(
            sp.sin(x), (x, -10, 10), adaptive=False, n=1_000
        ).get_data,
        upstream.LineOver1DRangeSeries(
            sp.sin(x), (x, -10, 10), adaptive=False, n=1_000
        ).get_data,
    )

    yield (
        "simple line, 1M",
        msp.LineOver1DRangeSeries(
            x**2 + 2 * x - 1, (x, -10, 10), adaptive=False, n=1_000_000
        ).get_data,
        upstream.LineOver1DRangeSeries(
            x**2 + 2 * x - 1, (x, -10, 10), adaptive=False, n=1_000_000
        ).get_data,
    )

    expression = (
        sp.sin(x) * sp.exp(-(x**2) / 25)
        + sp.cos(3 * x) / (1 + x**2)
        + sp.log(x + 11)
        + sp.tanh(x / 3)
    )
    yield (
        "complex line, 1M",
        msp.LineOver1DRangeSeries(
            expression, (x, -10, 10), adaptive=False, n=1_000_000
        ).get_data,
        upstream.LineOver1DRangeSeries(
            expression, (x, -10, 10), adaptive=False, n=1_000_000
        ).get_data,
    )

    piecewise = sp.Piecewise(
        (sp.sin(x) * sp.exp(x / 5), x < 0),
        (sp.log(1 + x) + sp.sqrt(x), True),
    )
    yield (
        "piecewise line, 1M",
        msp.LineOver1DRangeSeries(
            piecewise, (x, -5, 10), adaptive=False, n=1_000_000
        ).get_data,
        upstream.LineOver1DRangeSeries(
            piecewise, (x, -5, 10), adaptive=False, n=1_000_000
        ).get_data,
    )

    expressions = (
        sp.cos(t) * (1 + t / 20),
        sp.sin(t) * (1 + t / 20),
        sp.sin(3 * t) / (1 + t / 10),
    )
    yield (
        "parametric 3D line, 1M",
        msp.Parametric3DLineSeries(
            *expressions, (t, 0, 40), n=1_000_000
        ).get_data,
        upstream.Parametric3DLineSeries(
            *expressions, (t, 0, 40), n=1_000_000
        ).get_data,
    )

    surface = (
        sp.sin(x * y)
        + sp.cos(x + y)
        + sp.exp(-(x**2 + y**2) / 10)
        + sp.tanh(x - y)
    )
    yield (
        "surface, 1200x1200",
        msp.SurfaceOver2DRangeSeries(
            surface, (x, -4, 4), (y, -4, 4), n1=1200, n2=1200
        ).get_data,
        upstream.SurfaceOver2DRangeSeries(
            surface, (x, -4, 4), (y, -4, 4), n1=1200, n2=1200
        ).get_data,
    )

    yield (
        "simple surface, 1200x1200",
        msp.SurfaceOver2DRangeSeries(
            x + y, (x, -4, 4), (y, -4, 4), n1=1200, n2=1200
        ).get_data,
        upstream.SurfaceOver2DRangeSeries(
            x + y, (x, -4, 4), (y, -4, 4), n1=1200, n2=1200
        ).get_data,
    )


def main():
    print(f"Machine: {cpu_name()}, {os.cpu_count()} logical CPUs, {platform.system()} {platform.machine()}")
    print()
    print("| case | mojo-sympy-plot | SymPy | ratio | result |")
    print("| --- | ---: | ---: | ---: | --- |")
    filters = tuple(
        part.strip().lower()
        for part in os.environ.get("BENCH_FILTER", "").split(",")
        if part.strip()
    )
    for name, ours, theirs in cases():
        if filters and not any(part in name.lower() for part in filters):
            continue
        ours()
        theirs()
        mojo_seconds = timeit(ours)
        sympy_seconds = timeit(theirs)
        ratio = sympy_seconds / mojo_seconds
        result = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{sympy_seconds * 1e3:.2f} ms | {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
