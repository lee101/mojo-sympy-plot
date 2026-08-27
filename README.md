# mojo-sympy-plot

SymPy plotting sample evaluation with a fused Mojo numerical kernel.

There is no PyPI or conda package named `sympy-plot`. The plotting implementation
identified by this target is part of SymPy itself, under `sympy.plotting`. This port
therefore compares against the real `sympy.plotting.series` classes from SymPy 1.14
and provides counterparts with the same constructor signatures and `get_data()`
result layouts for the covered uniform-sampling subset.

## Coverage

The following classes evaluate supported real-valued SymPy expressions in Mojo:

- `LineOver1DRangeSeries`
- `Parametric2DLineSeries`
- `Parametric3DLineSeries`
- `SurfaceOver2DRangeSeries`
- `ParametricSurfaceSeries`
- `ContourSeries`
- `ImplicitSeries` with uniform meshing

The expression compiler covers arithmetic, powers, modulo, trigonometric and
hyperbolic functions and their inverses, exponentials, logarithms, absolute value,
sign, floor, ceiling, `Min`, `Max`, `Piecewise`, `sinc`, comparisons, and Boolean
conditions used by piecewise expressions. Linear and logarithmic ranges, integer
ranges, numeric parameters, output transforms, step lines, numerical pole detection,
and line exclusions are supported.

Not covered are adaptive recursive sampling, complex-valued coordinate modes,
compound Boolean implicit regions, special functions such as gamma and Bessel
functions, symbolic pole detection, color-function evaluation for surfaces, and
SymPy's rendering backends. Unsupported expression nodes raise
`UnsupportedExpression`; unsupported sampling modes raise `NotImplementedError`.
This project produces numerical plotting samples, not figures.

## Install and build

```bash
pixi install
pixi run build
pixi run test
```

The build creates `dist/libmojo-sympy-plot.so`. The Python wrapper also builds it on
first use if it is missing. Set `MOJO_SYMPY_PLOT_LIB` to load an already-built shared
library from another location.

## Usage

```python
import sympy as sp
from mojo_sympy_plot import LineOver1DRangeSeries, SurfaceOver2DRangeSeries

x, y = sp.symbols("x y")

line = LineOver1DRangeSeries(
    sp.sin(x) * sp.exp(-x**2 / 10),
    (x, -8, 8),
    adaptive=False,
    n=100_000,
)
line_x, line_y = line.get_data()

surface = SurfaceOver2DRangeSeries(
    sp.sin(x * y) + sp.exp(-x**2 - y**2),
    (x, -3, 3),
    (y, -3, 3),
    n1=500,
    n2=500,
)
mesh_x, mesh_y, mesh_z = surface.get_data()
```

The shorter helpers return the same arrays:

```python
from mojo_sympy_plot import sample_line, sample_surface

line_x, line_y = sample_line(sp.sin(x), (x, 0, 2 * sp.pi), n=10_000)
mesh_x, mesh_y, mesh_z = sample_surface(
    sp.cos(x) * sp.sin(y), (x, -2, 2), (y, -2, 2), n1=300, n2=300
)
```

`sympy_plot` is also provided as a compatibility import and re-exports the same API.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz with 72
logical CPUs, Linux x86-64. Times are the best of two warmed runs. Each row constructs the
same series once, then times `get_data()` against SymPy's corresponding series on
the same expression, range, and sample count.

| case | mojo-sympy-plot | SymPy | ratio | result |
| --- | ---: | ---: | ---: | --- |
| identity line, 1M | 12.38 ms | 300.41 ms | 24.27x | faster |
| default sine line, 1k | 0.08 ms | 1.02 ms | 12.31x | faster |
| simple line, 1M | 30.17 ms | 646.94 ms | 21.44x | faster |
| complex line, 1M | 140.58 ms | 2695.65 ms | 19.18x | faster |
| piecewise line, 1M | 89.33 ms | 21633.19 ms | 242.16x | faster |
| parametric 3D line, 1M | 143.71 ms | 4375.71 ms | 30.45x | faster |
| surface, 1200x1200 | 193.40 ms | 4046.19 ms | 20.92x | faster |
| simple surface, 1200x1200 | 25.57 ms | 1031.39 ms | 40.34x | faster |

These results particularly favor fused evaluation. SymPy's NumPy path creates
intermediate arrays and converts evaluated results through complex arrays so it can
mask non-real values. The Mojo evaluator keeps each expression's intermediate values
in SIMD registers and writes one final float64 array. Results will vary by expression,
sample count, CPU, and thread count; simple plots with only a few hundred points are
dominated by Python and FFI overhead.

No GPU path is included. Every measured CPU case is already more than 5x faster
than upstream, so none is an optimization target; adding GPU transfers, allocation,
and dispatch to these kernels is not justified by the measured comparison.

## How it works

Python walks a SymPy expression and lowers it to a small, validated postfix bytecode.
NumPy creates the plotting domains with the same `linspace`, `geomspace`, and
row-major `meshgrid` layout used by SymPy. The bytecode, constants, input grids, and
output array cross a C ABI through `ctypes`; buffers cross as integer addresses.
Python holds every contiguous float64 NumPy buffer alive for the complete synchronous
call, and Mojo validates addresses, lengths, bytecode operands, and stack depth before
dereferencing them.

One Mojo kernel interprets the bytecode in SIMD-width chunks. Each chunk uses a fixed
64-entry register stack, so arithmetic and function chains are fused without
allocating temporary arrays. A specialized SIMD path handles the common
`unary(variable)` shape without interpreter-stack setup. Both paths use a scalar tail,
and grids of at least 262,144 elements are divided across physical CPU cores.
Compiled programs are reused until expressions, variables, or parameters change;
small line domains keep a private immutable template and return a fresh array copy.
The exported function is non-parametric and uses `AnyOrigin[mut=True]` pointers
reconstructed from the integer addresses.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The parity suite asserts numerical and behavioral agreement with SymPy's actual
plotting series and separately tests the expression engine against `lambdify`.

MIT licensed.
