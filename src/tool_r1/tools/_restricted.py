"""A small, bounded Python-syntax calculator, never a host Python executor.

There is deliberately no eval/exec, attribute access, import, file access,
network access, user-defined function, while loop, or mutation of containers.
The interpreter is a convenience for reproducible local tests, not a VM and
not a claim that arbitrary Python can be sandboxed inside the host process.
"""

from __future__ import annotations

import ast
import math
import time


class RestrictedError(ValueError):
    """Invalid code or a resource bound was reached."""


MAX_SOURCE_BYTES = 16_384
MAX_AST_NODES = 2_000
MAX_AST_DEPTH = 48
MAX_STEPS = 30_000
MAX_ITEMS = 4_096
MAX_INTEGER_BITS = 4_096
MAX_VALUE_DEPTH = 24
MAX_STRING_CHARS = 16_384
MAX_EXPONENT = 1_024


class Interpreter:
    def __init__(self, timeout: float = 5.0, max_output_chars: int = 4096):
        self.deadline = time.monotonic() + timeout
        self.max_output_chars = max_output_chars
        self.steps = 0
        self.output: list[str] = []
        self.output_chars = 0
        self.variables: dict[str, object] = {}

    def tick(self, amount: int = 1):
        self.steps += amount
        if self.steps > MAX_STEPS:
            raise RestrictedError("execution step limit exceeded")
        if time.monotonic() > self.deadline:
            raise RestrictedError("execution timeout exceeded")

    def check_value(self, value):
        """Bound the entire expanded value, including aliases and nesting."""
        stack = [(value, 0)]
        count = 0
        display_size = 0
        while stack:
            self.tick()
            current, depth = stack.pop()
            count += 1
            if count > MAX_ITEMS or depth > MAX_VALUE_DEPTH:
                raise RestrictedError("collection size or nesting limit exceeded")
            if type(current) is int:
                if current.bit_length() > MAX_INTEGER_BITS:
                    raise RestrictedError("integer bit limit exceeded")
                display_size += current.bit_length() // 3 + 3
            elif type(current) is float:
                if not math.isfinite(current):
                    raise RestrictedError("non-finite numbers are not supported")
            elif type(current) is str:
                if len(current) > MAX_STRING_CHARS:
                    raise RestrictedError("string size limit exceeded")
                display_size += len(current) * 4 + 2
            elif type(current) in (list, tuple):
                if len(current) > MAX_ITEMS:
                    raise RestrictedError("collection size limit exceeded")
                stack.extend((item, depth + 1) for item in current)
            elif type(current) is dict:
                if len(current) * 2 > MAX_ITEMS:
                    raise RestrictedError("collection size limit exceeded")
                for key, item in current.items():
                    stack.extend(((key, depth + 1), (item, depth + 1)))
            elif type(current) is range:
                if len(current) > MAX_ITEMS:
                    raise RestrictedError("range size limit exceeded")
            elif current is not None and type(current) is not bool:
                raise RestrictedError("unsupported value type")
            if display_size > 4 * MAX_STRING_CHARS:
                raise RestrictedError("expanded value size limit exceeded")
        return value

    def run(self, code: str) -> str:
        if len(code.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise RestrictedError("source size limit exceeded")
        try:
            tree = ast.parse(code, mode="exec")
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise RestrictedError("invalid Python syntax") from exc
        stack = [(tree, 0)]
        nodes = 0
        while stack:
            node, depth = stack.pop()
            nodes += 1
            if nodes > MAX_AST_NODES or depth > MAX_AST_DEPTH:
                raise RestrictedError("syntax size or depth limit exceeded")
            if type(node) not in ALLOWED_AST_TYPES:
                raise RestrictedError("unsupported syntax: " + type(node).__name__)
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise RestrictedError("private names are not supported")
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in BUILTINS):
                raise RestrictedError("only explicitly supported builtin calls are allowed")
            stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        self.block(tree.body)
        return "".join(self.output)

    def block(self, statements):
        for node in statements:
            self.statement(node)

    def statement(self, node):
        self.tick()
        if isinstance(node, ast.Expr):
            self.expression(node.value)
        elif isinstance(node, ast.Assign):
            value = self.expression(node.value)
            for target in node.targets:
                self.assign(target, value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            previous = self.expression(node.target)
            self.assign(node.target, self.binary(node.op, previous, self.expression(node.value)))
        elif isinstance(node, ast.If):
            self.block(node.body if self.expression(node.test) else node.orelse)
        elif isinstance(node, ast.For):
            values = self.iterable(self.expression(node.iter))
            for value in values:
                self.tick()
                self.assign(node.target, value)
                self.block(node.body)
            self.block(node.orelse)
        elif isinstance(node, ast.Pass):
            return
        else:
            raise RestrictedError("unsupported statement: " + type(node).__name__)

    def assign(self, target, value):
        self.tick()
        if isinstance(target, ast.Name) and not target.id.startswith("_"):
            if target.id in BUILTINS:
                raise RestrictedError("cannot shadow a supported builtin")
            self.variables[target.id] = self.check_value(value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            values = self.iterable(value)
            if len(target.elts) != len(values):
                raise RestrictedError("unpacking size mismatch")
            for part, item in zip(target.elts, values):
                self.assign(part, item)
        else:
            raise RestrictedError("only names and tuple unpacking may be assigned")

    def iterable(self, value):
        if type(value) not in (list, tuple, dict, str, range):
            raise RestrictedError("value is not a supported iterable")
        if len(value) > MAX_ITEMS:
            raise RestrictedError("iterable size limit exceeded")
        return value

    def expression(self, node):
        self.tick()
        if isinstance(node, ast.Constant):
            return self.check_value(node.value)
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id not in self.variables:
                raise RestrictedError("unknown variable: " + node.id)
            return self.variables[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [self.expression(item) for item in node.elts]
            return self.check_value(tuple(values) if isinstance(node, ast.Tuple) else values)
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise RestrictedError("dictionary unpacking is not supported")
            return self.check_value({self.expression(k): self.expression(v) for k, v in zip(node.keys, node.values)})
        if isinstance(node, ast.BinOp):
            return self.binary(node.op, self.expression(node.left), self.expression(node.right))
        if isinstance(node, ast.UnaryOp):
            value = self.expression(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            self.numeric(value)
            if isinstance(node.op, ast.USub):
                return self.check_value(-value)
            if isinstance(node.op, ast.UAdd):
                return value
            raise RestrictedError("unsupported unary operator")
        if isinstance(node, ast.BoolOp):
            result = self.expression(node.values[0])
            for item in node.values[1:]:
                if isinstance(node.op, ast.And) and not result:
                    break
                if isinstance(node.op, ast.Or) and result:
                    break
                result = self.expression(item)
            return result
        if isinstance(node, ast.Compare):
            previous = self.expression(node.left)
            for op, right in zip(node.ops, node.comparators):
                value = self.expression(right)
                if not self.compare(op, previous, value):
                    return False
                previous = value
            return True
        if isinstance(node, ast.IfExp):
            return self.expression(node.body if self.expression(node.test) else node.orelse)
        if isinstance(node, ast.Subscript):
            value = self.expression(node.value)
            if type(value) not in (list, tuple, dict, str, range):
                raise RestrictedError("unsupported subscript value")
            if isinstance(node.slice, ast.Slice):
                args = [self.expression(v) if v is not None else None for v in (node.slice.lower, node.slice.upper, node.slice.step)]
                if any(v is not None and type(v) is not int for v in args):
                    raise RestrictedError("slice indices must be integers")
                index = slice(*args)
            else:
                index = self.expression(node.slice)
            return self.check_value(value[index])
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            return self.comprehension(node)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in BUILTINS:
                raise RestrictedError("only explicitly supported builtin calls are allowed")
            if any(keyword.arg is None for keyword in node.keywords):
                raise RestrictedError("keyword unpacking is not supported")
            args = [self.expression(arg) for arg in node.args]
            kwargs = {keyword.arg: self.expression(keyword.value) for keyword in node.keywords}
            return self.call(node.func.id, args, kwargs)
        raise RestrictedError("unsupported expression: " + type(node).__name__)

    def numeric(self, value):
        if type(value) not in (int, float, bool):
            raise RestrictedError("operator requires numbers")

    def binary(self, op, left, right):
        self.tick()
        if isinstance(op, ast.Add) and type(left) in (str, list, tuple) and type(right) is type(left):
            limit = MAX_STRING_CHARS if type(left) is str else MAX_ITEMS
            if len(left) + len(right) > limit:
                raise RestrictedError("concatenation size limit exceeded")
            return self.check_value(left + right)
        if isinstance(op, ast.Mult) and (type(left) in (str, list, tuple) or type(right) in (str, list, tuple)):
            sequence, count = (left, right) if type(left) in (str, list, tuple) else (right, left)
            if type(count) is not int:
                raise RestrictedError("repetition count must be an integer")
            limit = MAX_STRING_CHARS if type(sequence) is str else MAX_ITEMS
            if len(sequence) * max(count, 0) > limit:
                raise RestrictedError("repetition size limit exceeded")
            return self.check_value(sequence * count)
        self.numeric(left)
        self.numeric(right)
        if isinstance(op, ast.Add):
            result = left + right
        elif isinstance(op, ast.Sub):
            result = left - right
        elif isinstance(op, ast.Mult):
            if type(left) is int and type(right) is int and left.bit_length() + right.bit_length() > MAX_INTEGER_BITS + 1:
                raise RestrictedError("integer multiplication size limit exceeded")
            result = left * right
        elif isinstance(op, ast.Div):
            result = left / right
        elif isinstance(op, ast.FloorDiv):
            result = left // right
        elif isinstance(op, ast.Mod):
            result = left % right
        elif isinstance(op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise RestrictedError("exponent limit exceeded")
            if type(left) is int and type(right) is int and right > 0 and max(1, abs(left).bit_length()) * right > MAX_INTEGER_BITS:
                raise RestrictedError("integer exponent size limit exceeded")
            result = left ** right
        else:
            raise RestrictedError("unsupported binary operator")
        return self.check_value(result)

    def compare(self, op, left, right):
        self.tick()
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.In):
            return left in self.iterable(right)
        if isinstance(op, ast.NotIn):
            return left not in self.iterable(right)
        raise RestrictedError("unsupported comparison operator")

    def comprehension(self, node):
        saved = self.variables.copy()
        result = []

        def collect(index):
            self.tick()
            if index == len(node.generators):
                if len(result) >= MAX_ITEMS:
                    raise RestrictedError("comprehension size limit exceeded")
                result.append(self.expression(node.elt))
                return
            generator = node.generators[index]
            if generator.is_async:
                raise RestrictedError("async comprehensions are not supported")
            for item in self.iterable(self.expression(generator.iter)):
                self.assign(generator.target, item)
                if all(self.expression(condition) for condition in generator.ifs):
                    collect(index + 1)
        try:
            collect(0)
        finally:
            self.variables = saved
        return self.check_value(result)

    def call(self, name, args, kwargs):
        self.tick()
        if kwargs and name not in ("print", "sorted", "round"):
            raise RestrictedError("keyword arguments are not supported for " + name)
        if name == "print":
            if any(key not in ("sep", "end") for key in kwargs):
                raise RestrictedError("unsupported print keyword")
            sep, end = kwargs.get("sep", " "), kwargs.get("end", "\n")
            if type(sep) is not str or type(end) is not str:
                raise RestrictedError("print sep and end must be strings")
            chunks = []
            size = len(end)
            for index, arg in enumerate(args):
                self.check_value(arg)
                chunk = str(arg)
                size += len(chunk) + (len(sep) if index else 0)
                if self.output_chars + size > self.max_output_chars:
                    raise RestrictedError("output size limit exceeded")
                chunks.append(chunk)
            if self.output_chars + size > self.max_output_chars:
                raise RestrictedError("output size limit exceeded")
            text = sep.join(chunks) + end
            self.output.append(text)
            self.output_chars += len(text)
            return None
        if name == "range":
            if not 1 <= len(args) <= 3 or any(type(arg) is not int for arg in args):
                raise RestrictedError("range requires one to three integers")
            return self.check_value(range(*args))
        if name in ("zip", "enumerate"):
            if name == "enumerate":
                if not 1 <= len(args) <= 2 or (len(args) == 2 and type(args[1]) is not int):
                    raise RestrictedError("invalid enumerate arguments")
                values = self.iterable(args[0])
                if len(values) * 3 + 1 > MAX_ITEMS:
                    raise RestrictedError("enumerate size limit exceeded")
                result = list(enumerate(values, args[1] if len(args) == 2 else 0))
            else:
                values = [self.iterable(arg) for arg in args]
                if values and min(len(v) for v in values) * (len(values) + 1) + 1 > MAX_ITEMS:
                    raise RestrictedError("zip size limit exceeded")
                result = list(zip(*values))
            return self.check_value(result)
        if name == "sum":
            if not 1 <= len(args) <= 2:
                raise RestrictedError("sum requires iterable and optional start")
            result = args[1] if len(args) == 2 else 0
            self.numeric(result)
            for value in self.iterable(args[0]):
                self.numeric(value)
                result = self.binary(ast.Add(), result, value)
            return result
        if name in ("min", "max"):
            values = self.iterable(args[0]) if len(args) == 1 else args
            self.tick(len(values))
            return self.check_value((min if name == "min" else max)(values))
        if name == "sorted":
            if len(args) != 1 or any(key != "reverse" for key in kwargs):
                raise RestrictedError("sorted requires one iterable and optional reverse")
            reverse = kwargs.get("reverse", False)
            if type(reverse) is not bool:
                raise RestrictedError("reverse must be a boolean")
            values = self.iterable(args[0])
            self.tick(len(values))
            return self.check_value(sorted(values, reverse=reverse))
        if name == "len":
            if len(args) != 1:
                raise RestrictedError("len requires one argument")
            return len(self.iterable(args[0]))
        if name == "abs":
            if len(args) != 1:
                raise RestrictedError("abs requires one argument")
            self.numeric(args[0])
            return self.check_value(abs(args[0]))
        if name == "round":
            if not 1 <= len(args) <= 2 or any(key != "ndigits" for key in kwargs) or (len(args) == 2 and kwargs):
                raise RestrictedError("invalid round arguments")
            self.numeric(args[0])
            digits = args[1] if len(args) == 2 else kwargs.get("ndigits", None)
            if digits is not None and (type(digits) is not int or abs(digits) > MAX_EXPONENT):
                raise RestrictedError("round precision limit exceeded")
            return self.check_value(round(args[0], digits))
        raise RestrictedError("unsupported builtin")


BUILTINS = frozenset({"sum", "len", "min", "max", "sorted", "round", "range", "print", "abs", "enumerate", "zip"})

ALLOWED_AST_TYPES = frozenset({
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.If, ast.For, ast.Pass,
    ast.Constant, ast.Name, ast.Load, ast.Store, ast.List, ast.Tuple, ast.Dict,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Subscript,
    ast.Slice, ast.ListComp, ast.GeneratorExp, ast.comprehension, ast.Call,
    ast.keyword, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.Not, ast.USub, ast.UAdd, ast.And, ast.Or, ast.Eq, ast.NotEq,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
})
