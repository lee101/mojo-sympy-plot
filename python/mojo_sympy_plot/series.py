"""Mojo-backed uniform-sampling counterparts of SymPy plotting series."""

from __future__ import annotations

import numpy as np
import sympy as sp
from sympy.core.relational import Relational

from .compiler import compile_expression, evaluate_program

DOMAIN_CACHE_ELEMENTS = 16_384


def _number(value, params) -> float:
    if not params and isinstance(value, (int, float)):
        return float(value)
    result = sp.sympify(value)
    if params:
        result = result.subs(params)
    if result.free_symbols:
        raise ValueError(f"range endpoint is not numeric: {value}")
    return float(result)


def _domain(plot_range, count, scale, only_integers, params):
    symbol, start_expr, end_expr = plot_range
    start, end = _number(start_expr, params), _number(end_expr, params)
    if only_integers:
        start, end = int(start), int(end)
        return sp.sympify(symbol), np.arange(start, end + 1, dtype=np.float64)
    if count < 2:
        raise ValueError("sampling count must be at least 2")
    if scale == "linear":
        values = np.linspace(start, end, count, dtype=np.float64)
    elif scale == "log":
        values = np.geomspace(start, end, count, dtype=np.float64)
    else:
        raise ValueError("scale must be 'linear' or 'log'")
    return sp.sympify(symbol), values


def _apply(value, transform):
    return value if transform is None else np.asarray(transform(value))


class _Series:
    _N = 1000

    def __init__(self, **kwargs):
        n = kwargs.get("n", self._N)
        if hasattr(n, "__iter__") and not isinstance(n, (str, bytes)):
            ns = list(n)
            n = ns[0]
            n2 = ns[1] if len(ns) > 1 else self._N
        else:
            n2 = kwargs.get("n2", self._N)
        self._n = [int(kwargs.get("n1", n)), int(n2), int(kwargs.get("n3", self._N))]
        self._scales = [
            kwargs.get("xscale", "linear"),
            kwargs.get("yscale", "linear"),
            kwargs.get("zscale", "linear"),
        ]
        self.only_integers = bool(kwargs.get("only_integers", False))
        self._params = dict(kwargs.get("params", {}))
        self.rendering_kw = kwargs.get("rendering_kw", {})
        self._tx = kwargs.get("tx")
        self._ty = kwargs.get("ty")
        self._tz = kwargs.get("tz")
        self._tp = kwargs.get("tp")
        self.ranges = []
        self._compiled_key = None
        self._compiled_programs = ()
        self._line_domain_key = None
        self._line_domain_template = None

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        values = list(value) if hasattr(value, "__iter__") else [value]
        for i, item in enumerate(values[:3]):
            self._n[i] = int(item)

    @property
    def scales(self):
        return self._scales

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = dict(value)
        self._compiled_key = None
        self._compiled_programs = ()

    def _line_domain(self):
        if self.only_integers or self._n[0] > DOMAIN_CACHE_ELEMENTS:
            return _domain(
                self.ranges[0], self._n[0], self._scales[0],
                self.only_integers, self._params
            )
        key = (
            tuple(self.ranges[0]),
            self._n[0],
            self._scales[0],
            tuple(self._params.items()),
        )
        if key != self._line_domain_key:
            symbol, values = _domain(
                self.ranges[0], self._n[0], self._scales[0],
                False, self._params
            )
            values.setflags(write=False)
            self._line_domain_key = key
            self._line_domain_template = (symbol, values)
        symbol, values = self._line_domain_template
        return symbol, values.copy()

    def _surface_domain(self):
        sx, x = _domain(
            self.ranges[0], self._n[0], self._scales[0],
            self.only_integers, self._params
        )
        sy, y = _domain(
            self.ranges[1], self._n[1], self._scales[1],
            self.only_integers, self._params
        )
        mesh_x, mesh_y = np.meshgrid(x, y)
        return (sx, sy), (mesh_x, mesh_y)

    def _evaluate(self, expressions, variables, coordinates):
        key = (
            tuple(expressions),
            tuple(variables),
            tuple(self._params.items()),
        )
        if key != self._compiled_key:
            self._compiled_programs = tuple(
                compile_expression(expr, variables, params=self._params)
                for expr in expressions
            )
            self._compiled_key = key
        return [
            evaluate_program(program, *coordinates)
            for program in self._compiled_programs
        ]


class LineOver1DRangeSeries(_Series):
    """Covered counterpart of ``sympy.plotting.series.LineOver1DRangeSeries``."""

    def __init__(self, expr, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = sp.sympify(expr)
        self.ranges = [var_start_end]
        self._label = str(self.expr) if label is None else label
        self.adaptive = bool(kwargs.get("adaptive", False))
        self.steps = bool(kwargs.get("steps", False))
        self.detect_poles = kwargs.get("detect_poles", False)
        self.exclude = sorted(float(v) for v in kwargs.get("exclude", []))
        self._return = kwargs.get("return")

    def get_points(self):
        if self.adaptive and not self.only_integers:
            raise NotImplementedError("adaptive sampling is not covered")
        if self._return not in (None, "real"):
            raise NotImplementedError("complex-valued return modes are not covered")
        variable, x = self._line_domain()
        y = self._evaluate([self.expr], (variable,), (x,))[0]
        return x, y

    def get_data(self):
        x, y = self.get_points()
        x, y = _apply(x, self._tx), _apply(y, self._ty)
        if self.detect_poles:
            threshold = np.pi / 2 - float(getattr(self, "eps", 0.01))
            jumps = np.abs(np.arctan(np.abs(np.diff(y)) / np.diff(x))) >= threshold
            y = y.copy()
            y[1:][jumps] = np.nan
        if self.steps:
            x = np.array((x, x)).T.ravel()[1:]
            y = np.array((y, y)).T.ravel()[:-1]
        if self.exclude:
            variable = sp.sympify(self.ranges[0][0])
            for excluded in self.exclude:
                right = np.searchsorted(x, excluded)
                left = right - 1
                if left <= 0 or right >= x.size - 1:
                    continue
                delta = min(abs(excluded - x[left]), abs(excluded - x[right])) / 100
                inserted_x = np.array([excluded - delta, excluded, excluded + delta])
                inserted_y = self._evaluate([self.expr], (variable,), (inserted_x,))[0]
                inserted_y[1] = np.nan
                x = np.concatenate((x[:left], inserted_x, x[right:]))
                y = np.concatenate((y[:left], inserted_y, y[right:]))
        return x, y


class Parametric2DLineSeries(_Series):
    def __init__(self, expr_x, expr_y, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x, self.expr_y = sp.sympify(expr_x), sp.sympify(expr_y)
        self.expr = (self.expr_x, self.expr_y)
        self.ranges = [var_start_end]
        self._label = label
        self.adaptive = bool(kwargs.get("adaptive", False))
        self.color_func = kwargs.get("color_func")

    def get_data(self):
        if self.adaptive:
            raise NotImplementedError("adaptive sampling is not covered")
        variable, parameter = self._line_domain()
        x, y = self._evaluate(self.expr, (variable,), (parameter,))
        x, y, parameter = (
            _apply(x, self._tx), _apply(y, self._ty), _apply(parameter, self._tp)
        )
        color = (
            np.asarray(self.color_func(x, y, parameter))
            if callable(self.color_func) else parameter
        )
        return x, y, color

    get_points = get_data


class Parametric3DLineSeries(_Series):
    def __init__(self, expr_x, expr_y, expr_z, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x, self.expr_y, self.expr_z = map(
            sp.sympify, (expr_x, expr_y, expr_z)
        )
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end]
        self._label = label
        self.adaptive = bool(kwargs.get("adaptive", False))
        self.color_func = kwargs.get("color_func")

    def get_data(self):
        if self.adaptive:
            raise NotImplementedError("adaptive sampling is not covered")
        variable, parameter = self._line_domain()
        x, y, z = self._evaluate(self.expr, (variable,), (parameter,))
        x, y, z, parameter = (
            _apply(x, self._tx), _apply(y, self._ty),
            _apply(z, self._tz), _apply(parameter, self._tp)
        )
        color = (
            np.asarray(self.color_func(x, y, z, parameter))
            if callable(self.color_func) else parameter
        )
        return x, y, z, color

    get_points = get_data


class SurfaceOver2DRangeSeries(_Series):
    _N = 100

    def __init__(
        self, expr, var_start_end_x, var_start_end_y, label="", **kwargs
    ):
        super().__init__(**kwargs)
        self.expr = sp.sympify(expr)
        self.ranges = [var_start_end_x, var_start_end_y]
        self._label = str(self.expr) if label is None else label
        self.is_polar = bool(kwargs.get("is_polar", kwargs.get("polar", False)))

    def get_data(self):
        variables, meshes = self._surface_domain()
        z = self._evaluate([self.expr], variables, meshes)[0]
        x, y = meshes
        if self.is_polar:
            radius = x.copy()
            x, y = radius * np.cos(y), radius * np.sin(y)
        return _apply(x, self._tx), _apply(y, self._ty), _apply(z, self._tz)

    def get_meshes(self):
        return self.get_data()


class ParametricSurfaceSeries(_Series):
    _N = 100

    def __init__(
        self, expr_x, expr_y, expr_z, var_start_end_u, var_start_end_v,
        label="", **kwargs
    ):
        super().__init__(**kwargs)
        self.expr_x, self.expr_y, self.expr_z = map(
            sp.sympify, (expr_x, expr_y, expr_z)
        )
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end_u, var_start_end_v]
        self._label = label

    def get_data(self):
        variables, meshes = self._surface_domain()
        x, y, z = self._evaluate(self.expr, variables, meshes)
        u, v = meshes
        return (
            _apply(x, self._tx), _apply(y, self._ty), _apply(z, self._tz),
            u, v,
        )

    def get_meshes(self):
        return self.get_data()[:3]


class ContourSeries(SurfaceOver2DRangeSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_filled = kwargs.get("is_filled", kwargs.get("fill", True))
        self.show_clabels = kwargs.get("clabels", True)


class ImplicitSeries(_Series):
    _N = 100

    def __init__(
        self, expr, var_start_end_x, var_start_end_y, label="", **kwargs
    ):
        super().__init__(**kwargs)
        self.expr = sp.sympify(expr)
        self.ranges = [var_start_end_x, var_start_end_y]
        self._label = str(self.expr) if label is None else label
        self.adaptive = bool(kwargs.get("adaptive", False))

    def _normalized(self):
        expr = self.expr
        if isinstance(expr, sp.Equality):
            return expr.lhs - expr.rhs, "contour"
        if isinstance(expr, (sp.StrictGreaterThan, sp.GreaterThan)):
            return expr.lhs - expr.rhs, "contourf"
        if isinstance(expr, (sp.StrictLessThan, sp.LessThan)):
            return expr.rhs - expr.lhs, "contourf"
        if isinstance(expr, Relational):
            raise NotImplementedError(f"implicit relation {expr.func.__name__} is not covered")
        if expr.is_Boolean:
            raise NotImplementedError("compound Boolean implicit regions are not covered")
        return expr, "contour"

    def get_data(self):
        if self.adaptive:
            raise NotImplementedError("adaptive implicit sampling is not covered")
        expression, plot_type = self._normalized()
        variables, meshes = self._surface_domain()
        z = self._evaluate([expression], variables, meshes)[0]
        return meshes[0], meshes[1], z, plot_type
