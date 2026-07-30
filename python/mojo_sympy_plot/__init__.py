"""SymPy plotting sample evaluation accelerated by Mojo."""

from .compiler import (
    Program,
    UnsupportedExpression,
    compile_expression,
    evaluate,
    evaluate_program,
)
from .series import (
    ContourSeries,
    ImplicitSeries,
    LineOver1DRangeSeries,
    Parametric2DLineSeries,
    Parametric3DLineSeries,
    ParametricSurfaceSeries,
    SurfaceOver2DRangeSeries,
)

__version__ = "0.1.0"


def sample_line(expr, var_start_end, **kwargs):
    """Return ``(x, y)`` using the covered LineOver1DRangeSeries signature."""
    return LineOver1DRangeSeries(expr, var_start_end, **kwargs).get_data()


def sample_surface(expr, var_start_end_x, var_start_end_y, **kwargs):
    """Return ``(mesh_x, mesh_y, z)`` for a uniformly sampled surface."""
    return SurfaceOver2DRangeSeries(
        expr, var_start_end_x, var_start_end_y, **kwargs
    ).get_data()


__all__ = [
    "ContourSeries",
    "ImplicitSeries",
    "LineOver1DRangeSeries",
    "Parametric2DLineSeries",
    "Parametric3DLineSeries",
    "ParametricSurfaceSeries",
    "Program",
    "SurfaceOver2DRangeSeries",
    "UnsupportedExpression",
    "compile_expression",
    "evaluate",
    "evaluate_program",
    "sample_line",
    "sample_surface",
]
