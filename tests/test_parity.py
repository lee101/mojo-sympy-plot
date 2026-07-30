"""Numerical parity with SymPy's uniform plotting-series evaluator."""

import numpy as np
import pytest
import sympy as sp
from sympy.plotting import series as upstream

import mojo_sympy_plot as msp
from mojo_sympy_plot._lib import lib
from mojo_sympy_plot.compiler import compile_expression

x, y, t, u, v, a = sp.symbols("x y t u v a")


def assert_data_equal(ours, theirs, *, rtol=1e-8, atol=1e-10):
    assert len(ours) == len(theirs)
    for got, expected in zip(ours, theirs):
        if isinstance(expected, str):
            assert got == expected
        else:
            assert np.allclose(got, expected, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize(
    "expr",
    [
        sp.sin(x) + sp.cos(x) + sp.tan(x / 5),
        sp.asin(x / 2) + sp.acos(x / 2) + sp.atan(x),
        sp.sinh(x) + sp.cosh(x) + sp.tanh(x),
        sp.asinh(x) + sp.acosh(x + 2) + sp.atanh(x / 2),
        sp.exp(x) + sp.log(x + 2) + sp.log(x + 2, 10),
        sp.Abs(x) + sp.sign(x) + sp.floor(x) + sp.ceiling(x),
        sp.Min(x, x**2) + sp.Max(x, -x),
        sp.Mod(x, sp.Rational(3, 2)),
        sp.Piecewise((x**2, x < 0), (sp.sqrt(x), True)),
        sp.Piecewise(
            (sp.atan2(x, x + 1), sp.Or(x < -0.5, x > 0.5)),
            (sp.exp(x) - 1, sp.And(x >= -0.5, sp.Ne(x, 0.25))),
            (0, True),
        ),
        sp.sinc(x),
    ],
)
def test_expression_engine_matches_lambdify(expr):
    grid = np.linspace(-0.9, 0.9, 10_003)
    got = msp.evaluate(expr, (x,), grid)
    with np.errstate(all="ignore"):
        expected = np.asarray(sp.lambdify(x, expr, "numpy")(grid), dtype=np.float64)
    if expected.ndim == 0:
        expected = np.full_like(grid, expected)
    assert np.allclose(got, expected, rtol=1e-8, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("size", [7, 262_143, 262_147])
def test_unary_simd_tail_and_parallel_threshold(size):
    grid = np.linspace(-0.9, 0.9, size)
    got = msp.evaluate(sp.sin(x), (x,), grid)
    assert np.allclose(got, np.sin(grid), rtol=1e-8, atol=1e-10)


def test_two_variable_engine_matches_lambdify():
    gx, gy = np.meshgrid(np.linspace(-2, 2, 71), np.linspace(0.1, 3, 53))
    expr = sp.sin(x * y) + sp.exp(-x**2) / (1 + y**2)
    got = msp.evaluate(expr, (x, y), gx, gy)
    expected = sp.lambdify((x, y), expr, "numpy")(gx, gy)
    assert np.allclose(got, expected, rtol=1e-8, atol=1e-10)


def test_coordinate_validation_rejects_unsafe_narrowing_and_shape_mismatch():
    with pytest.raises(TypeError, match="real numeric"):
        msp.evaluate(x, (x,), np.array([1 + 2j]))
    with pytest.raises(TypeError, match="wider than float64"):
        msp.evaluate(x, (x,), np.array([1], dtype=np.longdouble))
    with pytest.raises(ValueError, match="exactly representable"):
        msp.evaluate(x, (x,), np.array([2**53 + 1], dtype=np.int64))
    with pytest.raises(ValueError, match="identical shapes"):
        msp.evaluate(x + y, (x, y), np.ones((2, 3)), np.ones((6,)))


def test_ffi_rejects_null_short_and_malformed_buffers():
    program = compile_expression(x, (x,))
    values = np.ones(4, dtype=np.float64)
    destination = np.empty_like(values)
    ffi = lib().msp_evaluate
    valid = (
        program.code_address, program.code.shape[0],
        program.constants_address, program.constants.size,
        values.ctypes.data, values.size, values.ctypes.data, values.size,
        1, destination.ctypes.data, destination.size, values.size, 1,
    )
    assert ffi(*((0,) + valid[1:])) == 1
    short = list(valid)
    short[5] = values.size - 1
    assert ffi(*short) == 1
    bad_code = np.array([[1, 2]], dtype=np.int64)
    malformed = list(valid)
    malformed[0] = bad_code.ctypes.data
    malformed[1] = 1
    assert ffi(*malformed) == 2


def test_line_series_matches_sympy():
    expr = sp.sin(x**2) * sp.exp(-x / 3) + sp.log(x + 4)
    kwargs = dict(adaptive=False, n=2001)
    ours = msp.LineOver1DRangeSeries(expr, (x, -3, 7), **kwargs)
    theirs = upstream.LineOver1DRangeSeries(expr, (x, -3, 7), **kwargs)
    assert_data_equal(ours.get_data(), theirs.get_data())
    assert_data_equal(ours.get_points(), theirs.get_points())


def test_line_log_scale_matches_sympy():
    kwargs = dict(adaptive=False, n=301, xscale="log")
    ours = msp.LineOver1DRangeSeries(sp.log(x), (x, 0.01, 100), **kwargs)
    theirs = upstream.LineOver1DRangeSeries(sp.log(x), (x, 0.01, 100), **kwargs)
    assert_data_equal(ours.get_data(), theirs.get_data())


def test_small_line_domain_cache_returns_fresh_arrays_and_tracks_n():
    series = msp.LineOver1DRangeSeries(
        sp.sin(x), (x, -1, 1), adaptive=False, n=1000
    )
    first_x, _ = series.get_data()
    first_x[0] = 99.0
    second_x, _ = series.get_data()
    assert second_x[0] == -1.0
    series.n = 1001
    resized_x, _ = series.get_data()
    assert resized_x.size == 1001


def test_only_integer_line_matches_sympy():
    kwargs = dict(adaptive=False, n=100, only_integers=True)
    ours = msp.LineOver1DRangeSeries(x**3 - 2 * x, (x, -7, 9), **kwargs)
    theirs = upstream.LineOver1DRangeSeries(x**3 - 2 * x, (x, -7, 9), **kwargs)
    assert_data_equal(ours.get_data(), theirs.get_data())


def test_parameter_update_matches_sympy():
    kwargs = dict(adaptive=False, n=101, params={a: 2.0})
    ours = msp.LineOver1DRangeSeries(sp.sin(a * x), (x, 0, a), **kwargs)
    theirs = upstream.LineOver1DRangeSeries(sp.sin(a * x), (x, 0, a), **kwargs)
    assert_data_equal(ours.get_data(), theirs.get_data())
    ours.params = {a: 3.5}
    theirs.params = {a: 3.5}
    assert_data_equal(ours.get_data(), theirs.get_data())


def test_in_place_parameter_update_invalidates_compiled_program():
    series = msp.LineOver1DRangeSeries(
        sp.sin(a * x), (x, 0, 2), adaptive=False, n=101, params={a: 2.0}
    )
    series.get_data()
    series.params[a] = 3.5
    got_x, got_y = series.get_data()
    assert np.allclose(got_y, np.sin(3.5 * got_x), rtol=1e-8, atol=1e-10)


def test_steps_and_transforms_match_sympy():
    kwargs = dict(
        adaptive=False,
        n=17,
        steps=True,
        tx=lambda q: q + 1,
        ty=lambda q: q * 2,
    )
    ours = msp.LineOver1DRangeSeries(x**2, (x, -1, 1), **kwargs)
    theirs = upstream.LineOver1DRangeSeries(x**2, (x, -1, 1), **kwargs)
    assert_data_equal(ours.get_data(), theirs.get_data())


def test_detect_poles_matches_sympy():
    kwargs = dict(adaptive=False, n=1000, detect_poles=True)
    ours = msp.LineOver1DRangeSeries(sp.tan(x), (x, -2, 2), **kwargs)
    theirs = upstream.LineOver1DRangeSeries(sp.tan(x), (x, -2, 2), **kwargs)
    assert_data_equal(ours.get_data(), theirs.get_data())


def test_line_exclusions_match_sympy():
    kwargs = dict(adaptive=False, n=101, exclude=[sp.Rational(1, 3)])
    ours = msp.LineOver1DRangeSeries(sp.sin(x), (x, -1, 1), **kwargs)
    theirs = upstream.LineOver1DRangeSeries(sp.sin(x), (x, -1, 1), **kwargs)
    assert_data_equal(ours.get_data(), theirs.get_data())


def test_parametric_2d_line_matches_sympy():
    kwargs = dict(adaptive=False, n=1501)
    ours = msp.Parametric2DLineSeries(sp.cos(t), sp.sin(2 * t), (t, 0, 2), **kwargs)
    theirs = upstream.Parametric2DLineSeries(
        sp.cos(t), sp.sin(2 * t), (t, 0, 2), **kwargs
    )
    assert_data_equal(ours.get_data(), theirs.get_data())


def test_parametric_3d_line_matches_sympy():
    kwargs = dict(n=1201)
    ours = msp.Parametric3DLineSeries(t, t**2, sp.sin(t), (t, -2, 2), **kwargs)
    theirs = upstream.Parametric3DLineSeries(t, t**2, sp.sin(t), (t, -2, 2), **kwargs)
    assert_data_equal(ours.get_data(), theirs.get_data())


def test_surface_matches_sympy():
    expr = sp.sin(x * y) + sp.exp(-x**2 - y**2)
    kwargs = dict(n1=81, n2=67)
    ours = msp.SurfaceOver2DRangeSeries(expr, (x, -2, 2), (y, -3, 3), **kwargs)
    theirs = upstream.SurfaceOver2DRangeSeries(expr, (x, -2, 2), (y, -3, 3), **kwargs)
    assert_data_equal(ours.get_data(), theirs.get_data())
    assert_data_equal(ours.get_meshes(), theirs.get_meshes())


def test_polar_surface_matches_sympy():
    kwargs = dict(n1=41, n2=31, polar=True)
    ours = msp.SurfaceOver2DRangeSeries(
        sp.sin(x), (x, 0.1, 2), (y, 0, sp.pi), **kwargs
    )
    theirs = upstream.SurfaceOver2DRangeSeries(
        sp.sin(x), (x, 0.1, 2), (y, 0, sp.pi), **kwargs
    )
    assert_data_equal(ours.get_data(), theirs.get_data())


def test_parametric_surface_matches_sympy():
    expressions = (sp.cos(u) * v, sp.sin(u) * v, u + v)
    ranges = ((u, 0, sp.pi), (v, 0.5, 2))
    kwargs = dict(n1=59, n2=47)
    ours = msp.ParametricSurfaceSeries(*expressions, *ranges, **kwargs)
    theirs = upstream.ParametricSurfaceSeries(*expressions, *ranges, **kwargs)
    assert_data_equal(ours.get_data(), theirs.get_data())
    assert_data_equal(ours.get_meshes(), theirs.get_meshes())


def test_surface_and_parametric_transforms_match_sympy():
    surface_kwargs = dict(n1=19, n2=17, tx=lambda q: q + 1, tz=lambda q: q * 2)
    ours = msp.SurfaceOver2DRangeSeries(
        x + y, (x, -1, 1), (y, -2, 2), **surface_kwargs
    )
    theirs = upstream.SurfaceOver2DRangeSeries(
        x + y, (x, -1, 1), (y, -2, 2), **surface_kwargs
    )
    assert_data_equal(ours.get_data(), theirs.get_data())

    parametric_kwargs = dict(
        n=31, tx=lambda q: q + 1, ty=lambda q: q - 1, tp=lambda q: q * 3
    )
    ours_line = msp.Parametric2DLineSeries(
        sp.cos(t), sp.sin(t), (t, 0, 1), **parametric_kwargs
    )
    theirs_line = upstream.Parametric2DLineSeries(
        sp.cos(t), sp.sin(t), (t, 0, 1), **parametric_kwargs
    )
    assert_data_equal(ours_line.get_data(), theirs_line.get_data())


def test_contour_matches_sympy():
    expr = sp.cos(x) + sp.sin(y) + x * y
    kwargs = dict(n1=73, n2=61)
    ours = msp.ContourSeries(expr, (x, -2, 2), (y, -3, 3), **kwargs)
    theirs = upstream.ContourSeries(expr, (x, -2, 2), (y, -3, 3), **kwargs)
    assert_data_equal(ours.get_data(), theirs.get_data())


@pytest.mark.parametrize(
    "expr",
    [
        sp.Eq(x**2 + y**2, 1),
        x + y > 0.5,
        x - y <= 0.25,
        x**2 + y**2 - 1,
    ],
)
def test_implicit_uniform_grid_matches_sympy(expr):
    kwargs = dict(adaptive=False, n1=79, n2=63)
    ours = msp.ImplicitSeries(expr, (x, -2, 2), (y, -2, 2), **kwargs)
    theirs = upstream.ImplicitSeries(expr, (x, -2, 2), (y, -2, 2), **kwargs)
    assert_data_equal(ours.get_data(), theirs.get_data())


def test_constant_expression_is_broadcast():
    ours = msp.LineOver1DRangeSeries(sp.pi, (x, -1, 1), adaptive=False, n=31)
    theirs = upstream.LineOver1DRangeSeries(sp.pi, (x, -1, 1), adaptive=False, n=31)
    assert_data_equal(ours.get_data(), theirs.get_data())


def test_real_domain_errors_are_nan_like_sympy():
    ours = msp.LineOver1DRangeSeries(sp.sqrt(x), (x, -1, 1), adaptive=False, n=101)
    theirs = upstream.LineOver1DRangeSeries(sp.sqrt(x), (x, -1, 1), adaptive=False, n=101)
    assert_data_equal(ours.get_data(), theirs.get_data())


def test_unsupported_expression_fails_explicitly():
    with pytest.raises(msp.UnsupportedExpression, match="gamma"):
        msp.sample_line(sp.gamma(x), (x, 1, 3), adaptive=False, n=20)


def test_adaptive_sampling_fails_explicitly():
    series = msp.LineOver1DRangeSeries(sp.sin(x), (x, 0, 1), adaptive=True)
    with pytest.raises(NotImplementedError, match="adaptive"):
        series.get_data()
    series_3d = msp.Parametric3DLineSeries(
        t, t**2, t**3, (t, 0, 1), adaptive=True
    )
    with pytest.raises(NotImplementedError, match="adaptive"):
        series_3d.get_data()


def test_compatibility_package_exports_same_class():
    import sympy_plot

    assert sympy_plot.LineOver1DRangeSeries is msp.LineOver1DRangeSeries
