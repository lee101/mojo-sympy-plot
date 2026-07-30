"""Lower real-valued SymPy expressions to the Mojo evaluator bytecode."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ._lib import addr, lib

CPU_COUNT = os.cpu_count() or 1

LOAD_VAR, LOAD_CONST = 1, 2
ADD, SUB, MUL, DIV, POW, MOD = range(10, 16)
NEG, ABS, SIGN = range(20, 23)
LT, LE, GT, GE, EQ, NE, AND, OR = range(30, 38)
(
    SIN,
    COS,
    TAN,
    ASIN,
    ACOS,
    ATAN,
    ATAN2,
    SINH,
    COSH,
    TANH,
    ASINH,
    ACOSH,
    ATANH,
    EXP,
    EXPM1,
    LOG,
    LOG1P,
    LOG10,
    SQRT,
    FLOOR,
    CEIL,
    MINIMUM,
    MAXIMUM,
    WHERE,
) = range(40, 64)

_UNARY = {
    sp.sin: SIN,
    sp.cos: COS,
    sp.tan: TAN,
    sp.asin: ASIN,
    sp.acos: ACOS,
    sp.atan: ATAN,
    sp.sinh: SINH,
    sp.cosh: COSH,
    sp.tanh: TANH,
    sp.asinh: ASINH,
    sp.acosh: ACOSH,
    sp.atanh: ATANH,
    sp.exp: EXP,
    sp.log: LOG,
    sp.Abs: ABS,
    sp.sign: SIGN,
    sp.floor: FLOOR,
    sp.ceiling: CEIL,
}
_RELATIONAL = {
    sp.StrictLessThan: LT,
    sp.LessThan: LE,
    sp.StrictGreaterThan: GT,
    sp.GreaterThan: GE,
    sp.Equality: EQ,
    sp.Unequality: NE,
}


class UnsupportedExpression(ValueError):
    pass


@dataclass(frozen=True)
class Program:
    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    code: np.ndarray
    constants: np.ndarray
    code_address: int = field(init=False, repr=False)
    constants_address: int = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "code_address", addr(self.code))
        object.__setattr__(self, "constants_address", addr(self.constants))


class _Compiler:
    def __init__(self, expression, variables):
        self.expression = sp.sympify(expression)
        self.variables = tuple(variables)
        self.variable_indexes = {symbol: i for i, symbol in enumerate(self.variables)}
        self.instructions: list[tuple[int, int]] = []
        self.constants: list[float] = []
        self.depth = 0
        self.max_depth = 0

    def emit(self, opcode: int, argument: int = 0, delta: int = 0) -> None:
        self.instructions.append((opcode, argument))
        self.depth += delta
        self.max_depth = max(self.max_depth, self.depth)

    def compile(self) -> Program:
        unknown = self.expression.free_symbols.difference(self.variables)
        if unknown:
            names = ", ".join(sorted(map(str, unknown)))
            raise UnsupportedExpression(f"unbound symbols: {names}")
        self.visit(self.expression)
        if self.depth != 1:
            raise UnsupportedExpression("invalid expression stack")
        if self.max_depth > 64:
            raise UnsupportedExpression("expression requires more than 64 stack values")
        if len(self.instructions) > 256:
            raise UnsupportedExpression("expression exceeds 256 bytecode operations")
        return Program(
            self.expression,
            self.variables,
            np.ascontiguousarray(self.instructions, dtype=np.int64),
            np.ascontiguousarray(self.constants or [0.0], dtype=np.float64),
        )

    def fold(self, arguments, opcode: int) -> None:
        self.visit(arguments[0])
        for argument in arguments[1:]:
            self.visit(argument)
            self.emit(opcode, delta=-1)

    def visit_piecewise(self, arguments, index: int = 0) -> None:
        if index >= len(arguments):
            self.visit(sp.nan)
            return
        expression, condition = arguments[index]
        if condition is sp.true:
            self.visit(expression)
            return
        self.visit(condition)
        self.visit(expression)
        self.visit_piecewise(arguments, index + 1)
        self.emit(WHERE, delta=-2)

    def visit(self, expression) -> None:
        if expression in (sp.true, sp.false):
            self.constants.append(float(expression is sp.true))
            self.emit(LOAD_CONST, len(self.constants) - 1, 1)
            return
        if isinstance(expression, sp.Symbol):
            if expression not in self.variable_indexes:
                raise UnsupportedExpression(f"unbound symbol: {expression}")
            self.emit(LOAD_VAR, self.variable_indexes[expression], 1)
            return
        if expression.is_number:
            if expression.is_real is False:
                raise UnsupportedExpression(f"complex constant is unsupported: {expression}")
            try:
                value = float(expression)
            except (TypeError, ValueError, OverflowError) as exc:
                raise UnsupportedExpression(
                    f"constant cannot be represented as float64: {expression}"
                ) from exc
            self.constants.append(value)
            self.emit(LOAD_CONST, len(self.constants) - 1, 1)
            return
        if isinstance(expression, sp.Add):
            self.fold(expression.args, ADD)
            return
        if isinstance(expression, sp.Mul):
            self.fold(expression.args, MUL)
            return
        if isinstance(expression, sp.Pow):
            self.fold(expression.args, POW)
            return
        if isinstance(expression, sp.Mod):
            self.fold(expression.args, MOD)
            return
        if isinstance(expression, sp.Piecewise):
            self.visit_piecewise(expression.args)
            return
        if isinstance(expression, sp.Min):
            self.fold(expression.args, MINIMUM)
            return
        if isinstance(expression, sp.Max):
            self.fold(expression.args, MAXIMUM)
            return
        if isinstance(expression, sp.And):
            self.fold(expression.args, AND)
            return
        if isinstance(expression, sp.Or):
            self.fold(expression.args, OR)
            return
        for relation_type, opcode in _RELATIONAL.items():
            if isinstance(expression, relation_type):
                self.visit(expression.lhs)
                self.visit(expression.rhs)
                self.emit(opcode, delta=-1)
                return
        if expression.func is sp.atan2 and len(expression.args) == 2:
            self.fold(expression.args, ATAN2)
            return
        if expression.func is sp.log:
            if len(expression.args) == 1:
                self.visit(expression.args[0])
                self.emit(LOG)
            elif len(expression.args) == 2:
                self.visit(expression.args[0])
                self.emit(LOG)
                self.visit(expression.args[1])
                self.emit(LOG)
                self.emit(DIV, delta=-1)
            else:
                raise UnsupportedExpression("log() arity is unsupported")
            return
        if expression.func is sp.sinc:
            argument = expression.args[0]
            self.visit(sp.Eq(argument, 0))
            self.visit(sp.Integer(1))
            self.visit(sp.sin(argument) / argument)
            self.emit(WHERE, delta=-2)
            return
        opcode = _UNARY.get(expression.func)
        if opcode is not None and len(expression.args) == 1:
            self.visit(expression.args[0])
            self.emit(opcode)
            return
        raise UnsupportedExpression(
            f"unsupported SymPy node {expression.func.__name__}: {expression}"
        )


def compile_expression(expression, variables, params=None) -> Program:
    variables = tuple(map(sp.sympify, variables))
    if not 1 <= len(variables) <= 2:
        raise ValueError("one or two plotting variables are required")
    expr = sp.sympify(expression)
    if params:
        expr = expr.subs(params)
    return _Compiler(expr, variables).compile()


def evaluate_program(program: Program, *coordinates: np.ndarray) -> np.ndarray:
    if len(coordinates) != len(program.variables):
        raise ValueError("coordinate count does not match the compiled expression")
    arrays = []
    for coordinate in coordinates:
        source = np.asarray(coordinate)
        if source.dtype.kind not in "biuf":
            raise TypeError("coordinates must contain real numeric values")
        if source.dtype.kind == "f" and source.dtype.itemsize > 8:
            raise TypeError("coordinates wider than float64 are not supported")
        if source.dtype.kind in "iu" and source.size:
            limit = 2**53
            if np.any(source > limit) or np.any(source < -limit):
                raise ValueError("integer coordinates must be exactly representable as float64")
        arrays.append(np.ascontiguousarray(source, dtype=np.float64))
    if not arrays or arrays[0].size == 0:
        raise ValueError("coordinate arrays must not be empty")
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("coordinate arrays must have identical shapes")
    destination = np.empty(shape, dtype=np.float64)
    x_address = addr(arrays[0])
    y_address = addr(arrays[1]) if len(arrays) == 2 else x_address
    status = lib().msp_evaluate(
        program.code_address,
        program.code.shape[0],
        program.constants_address,
        program.constants.size,
        x_address,
        arrays[0].size,
        y_address,
        arrays[1].size if len(arrays) == 2 else arrays[0].size,
        len(arrays),
        addr(destination),
        destination.size,
        destination.size,
        CPU_COUNT,
    )
    if status:
        raise RuntimeError(f"Mojo expression evaluator returned status {status}")
    return destination


def evaluate(expression, variables, *coordinates, params=None) -> np.ndarray:
    """Evaluate a real SymPy expression over one or two equal-shaped arrays."""
    return evaluate_program(
        compile_expression(expression, variables, params=params), *coordinates
    )
