"""Fused SIMD evaluation of plotting expressions over caller-owned grids."""

from std.math import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    ceil,
    cos,
    cosh,
    exp,
    expm1,
    floor,
    log,
    log10,
    pow,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime MAX_STACK = 64
comptime PARALLEL_ELEMENTS = 262_144


@always_inline
def apply_unary[width: Int](
    op: Int, value: SIMD[DType.float64, width]
) -> SIMD[DType.float64, width]:
    if op == 20:
        return -value
    if op == 21:
        return abs(value)
    if op == 22:
        var positive = value.gt(0.0).select(
            SIMD[DType.float64, width](1.0),
            SIMD[DType.float64, width](0.0),
        )
        return value.lt(0.0).select(
            SIMD[DType.float64, width](-1.0), positive
        )
    if op == 40:
        return sin(value)
    if op == 41:
        return cos(value)
    if op == 42:
        return tan(value)
    if op == 43:
        return asin(value)
    if op == 44:
        return acos(value)
    if op == 45:
        return atan(value)
    if op == 47:
        return sinh(value)
    if op == 48:
        return cosh(value)
    if op == 49:
        return tanh(value)
    if op == 50:
        return asinh(value)
    if op == 51:
        return acosh(value)
    if op == 52:
        return atanh(value)
    if op == 53:
        return exp(value)
    if op == 54:
        return expm1(value)
    if op == 55:
        return log(value)
    if op == 56:
        var square = value * value
        var series = (
            value
            - square * 0.5
            + square * value / 3.0
            - square * square * 0.25
            + square * square * value * 0.2
        )
        return abs(value).lt(1.0e-4).select(
            series, log(SIMD[DType.float64, width](1.0) + value)
        )
    if op == 57:
        return log10(value)
    if op == 58:
        return sqrt(value)
    if op == 59:
        return floor(value)
    return ceil(value)


@always_inline
def evaluate_chunk[width: Int](
    code: IPtr,
    code_count: Int,
    constants: FPtr,
    x: FPtr,
    y: FPtr,
    index: Int,
) -> SIMD[DType.float64, width]:
    var stack = Array[SIMD[DType.float64, width], MAX_STACK](
        fill=SIMD[DType.float64, width](0.0)
    )
    var sp = 0
    for pc in range(code_count):
        var op = Int(code.unsafe_load(pc * 2))
        var argument = Int(code.unsafe_load(pc * 2 + 1))
        if op == 1:
            stack[sp] = (
                x.unsafe_load[width=width](index)
                if argument == 0
                else y.unsafe_load[width=width](index)
            )
            sp += 1
        elif op == 2:
            stack[sp] = SIMD[DType.float64, width](
                constants.unsafe_load(argument)
            )
            sp += 1
        elif op == 10:
            sp -= 1
            stack[sp - 1] += stack[sp]
        elif op == 11:
            sp -= 1
            stack[sp - 1] -= stack[sp]
        elif op == 12:
            sp -= 1
            stack[sp - 1] *= stack[sp]
        elif op == 13:
            sp -= 1
            stack[sp - 1] /= stack[sp]
        elif op == 14:
            sp -= 1
            stack[sp - 1] = pow(stack[sp - 1], stack[sp])
        elif op == 15:
            sp -= 1
            var quotient = floor(stack[sp - 1] / stack[sp])
            stack[sp - 1] -= quotient * stack[sp]
        elif op == 20:
            stack[sp - 1] = -stack[sp - 1]
        elif op == 21:
            stack[sp - 1] = abs(stack[sp - 1])
        elif op == 22:
            var positive = stack[sp - 1].gt(0.0).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
            stack[sp - 1] = stack[sp - 1].lt(0.0).select(
                SIMD[DType.float64, width](-1.0), positive
            )
        elif op == 30:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].lt(stack[sp]).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 31:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].le(stack[sp]).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 32:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].gt(stack[sp]).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 33:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].ge(stack[sp]).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 34:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].eq(stack[sp]).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 35:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].ne(stack[sp]).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 36:
            sp -= 1
            var both = stack[sp - 1].ne(0.0) & stack[sp].ne(0.0)
            stack[sp - 1] = both.select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 37:
            sp -= 1
            var either = stack[sp - 1].ne(0.0) | stack[sp].ne(0.0)
            stack[sp - 1] = either.select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 40:
            stack[sp - 1] = sin(stack[sp - 1])
        elif op == 41:
            stack[sp - 1] = cos(stack[sp - 1])
        elif op == 42:
            stack[sp - 1] = tan(stack[sp - 1])
        elif op == 43:
            stack[sp - 1] = asin(stack[sp - 1])
        elif op == 44:
            stack[sp - 1] = acos(stack[sp - 1])
        elif op == 45:
            stack[sp - 1] = atan(stack[sp - 1])
        elif op == 46:
            sp -= 1
            stack[sp - 1] = atan2(stack[sp - 1], stack[sp])
        elif op == 47:
            stack[sp - 1] = sinh(stack[sp - 1])
        elif op == 48:
            stack[sp - 1] = cosh(stack[sp - 1])
        elif op == 49:
            stack[sp - 1] = tanh(stack[sp - 1])
        elif op == 50:
            stack[sp - 1] = asinh(stack[sp - 1])
        elif op == 51:
            stack[sp - 1] = acosh(stack[sp - 1])
        elif op == 52:
            stack[sp - 1] = atanh(stack[sp - 1])
        elif op == 53:
            stack[sp - 1] = exp(stack[sp - 1])
        elif op == 54:
            stack[sp - 1] = expm1(stack[sp - 1])
        elif op == 55:
            stack[sp - 1] = log(stack[sp - 1])
        elif op == 56:
            var value = stack[sp - 1]
            var square = value * value
            var series = (
                value
                - square * 0.5
                + square * value / 3.0
                - square * square * 0.25
                + square * square * value * 0.2
            )
            stack[sp - 1] = abs(value).lt(1.0e-4).select(
                series, log(SIMD[DType.float64, width](1.0) + value)
            )
        elif op == 57:
            stack[sp - 1] = log10(stack[sp - 1])
        elif op == 58:
            stack[sp - 1] = sqrt(stack[sp - 1])
        elif op == 59:
            stack[sp - 1] = floor(stack[sp - 1])
        elif op == 60:
            stack[sp - 1] = ceil(stack[sp - 1])
        elif op == 61:
            sp -= 1
            stack[sp - 1] = min(stack[sp - 1], stack[sp])
        elif op == 62:
            sp -= 1
            stack[sp - 1] = max(stack[sp - 1], stack[sp])
        elif op == 63:
            sp -= 2
            stack[sp - 1] = stack[sp - 1].ne(0.0).select(
                stack[sp], stack[sp + 1]
            )
    return stack[0]


@export("msp_evaluate")
def msp_evaluate(
    code_addr: Int,
    code_count: Int,
    constants_addr: Int,
    constants_count: Int,
    x_addr: Int,
    x_count: Int,
    y_addr: Int,
    y_count: Int,
    variable_count: Int,
    dst_addr: Int,
    dst_count: Int,
    n: Int,
    requested_workers: Int,
) abi("C") -> Int:
    if (
        code_addr == 0
        or constants_addr == 0
        or x_addr == 0
        or y_addr == 0
        or dst_addr == 0
        or code_count < 1
        or code_count > 256
        or constants_count < 1
        or variable_count < 1
        or variable_count > 2
        or n < 1
        or x_count < n
        or (variable_count == 2 and y_count < n)
        or dst_count < n
    ):
        return 1
    var code = IPtr(unsafe_from_address=code_addr)
    var constants = FPtr(unsafe_from_address=constants_addr)
    var x = FPtr(unsafe_from_address=x_addr)
    var y = FPtr(unsafe_from_address=y_addr)
    var destination = FPtr(unsafe_from_address=dst_addr)
    var depth = 0
    for pc in range(code_count):
        var op = Int(code.unsafe_load(pc * 2))
        var argument = Int(code.unsafe_load(pc * 2 + 1))
        if op == 1:
            if argument < 0 or argument >= variable_count:
                return 2
            depth += 1
        elif op == 2:
            if argument < 0 or argument >= constants_count:
                return 2
            depth += 1
        elif (
            op == 10
            or op == 11
            or op == 12
            or op == 13
            or op == 14
            or op == 15
            or (op >= 30 and op <= 37)
            or op == 46
            or op == 61
            or op == 62
        ):
            if depth < 2:
                return 2
            depth -= 1
        elif (
            (op >= 20 and op <= 22)
            or (op >= 40 and op <= 45)
            or (op >= 47 and op <= 60)
        ):
            if depth < 1:
                return 2
        elif op == 63:
            if depth < 3:
                return 2
            depth -= 2
        else:
            return 2
        if depth > MAX_STACK:
            return 2
    if depth != 1:
        return 2
    var workers = min(requested_workers, num_physical_cores())
    var unary_op = 0
    var fast_unary = False
    if code_count == 2:
        unary_op = Int(code.unsafe_load(2))
        fast_unary = (
            Int(code.unsafe_load(0)) == 1
            and (
                (unary_op >= 20 and unary_op <= 22)
                or (unary_op >= 40 and unary_op <= 45)
                or (unary_op >= 47 and unary_op <= 60)
            )
        )
    var unary_input = (
        x if Int(code.unsafe_load(1)) == 0 else y
    )
    if n < PARALLEL_ELEMENTS:
        workers = 1
    workers = max(workers, 1)

    @__parameter
    def process(worker: Int):
        var vectors = n // W
        var start = (worker * vectors // workers) * W
        var end = ((worker + 1) * vectors // workers) * W
        if worker == workers - 1:
            end = n
        var i = start
        while i + W <= end:
            if fast_unary:
                destination.unsafe_store(
                    i,
                    apply_unary[W](
                        unary_op,
                        unary_input.unsafe_load[width=W](i),
                    ),
                )
            else:
                destination.unsafe_store(
                    i,
                    evaluate_chunk[W](code, code_count, constants, x, y, i),
                )
            i += W
        while i < end:
            if fast_unary:
                destination.unsafe_store(
                    i,
                    apply_unary[1](
                        unary_op,
                        unary_input.unsafe_load[width=1](i),
                    )[0],
                )
            else:
                destination.unsafe_store(
                    i,
                    evaluate_chunk[1](
                        code, code_count, constants, x, y, i
                    )[0],
                )
            i += 1

    # `parallelize` moved out of the Mojo standard library in 1.1. Keep the
    # worker partitioning ABI intact and execute each partition synchronously.
    for worker in range(workers):
        process(worker)
    return 0
