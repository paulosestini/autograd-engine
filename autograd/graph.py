import numpy as np

eps = 1e-12


class Node:
    def __init__(self) -> None:
        self.value: np.ndarray | None = None
        self.grad: np.ndarray | float = 0

    def eval(self) -> np.ndarray:
        raise NotImplementedError

    def backdiff(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        self.traverse_and_apply("_zero_grad")

    def _zero_grad(self) -> None:
        self.grad = 0

    def backward(self, init_grad: float | None = None) -> None:
        if init_grad is not None:
            self.grad = init_grad
        else:
            if self.value is None:
                raise RuntimeError("You must call .eval() before .backward().")
            if np.ndim(self.value) > 0 and np.prod(np.shape(self.value)) > 1:
                raise RuntimeError("Output is not scalar, pass init_grad explicitly.")
            self.grad = 1.0
        self.traverse_and_apply("backdiff")

    def traverse_and_apply(self, method: str) -> None:
        frontier = [self]
        while frontier:
            current = frontier.pop(0)
            if isinstance(current, UnaryOp):
                frontier.append(current.node)
            elif isinstance(current, BinaryOp):
                frontier.append(current.node1)
                frontier.append(current.node2)
            getattr(current, method)()

    def __add__(self, other: "Node | float") -> "AddOp":
        if not isinstance(other, Node):
            other = Value(other)
        return AddOp(self, other)

    def __radd__(self, other: float) -> "AddOp":
        return AddOp(Value(other), self)

    def __neg__(self) -> "NegOp":
        return NegOp(self)

    def __sub__(self, other: "Node | float") -> "AddOp":
        if not isinstance(other, Node):
            other = Value(other)
        return AddOp(self, NegOp(other))

    def __mul__(self, other: "Node | float") -> "MulOp":
        if not isinstance(other, Node):
            other = Value(other)
        return MulOp(self, other)

    def __rmul__(self, other: float) -> "MulOp":
        return MulOp(Value(other), self)

    def __matmul__(self, other: "Node") -> "MatMulOp":
        if not isinstance(other, Node):
            other = Value(other)
        return MatMulOp(self, other)

    def square(self) -> "SquareOp":
        return SquareOp(self)

    def sum(self) -> "SumOp":
        return SumOp(self)

    def softmax(self) -> "SoftmaxOp":
        return SoftmaxOp(self)

    def relu(self) -> "ReLUOp":
        return ReLUOp(self)

    def sigmoid(self) -> "SigmoidOp":
        return SigmoidOp(self)

    def cross_entropy(self, other: "Node") -> "CrossEntropyOp":
        return CrossEntropyOp(self, other)


class Value(Node):
    def __init__(self, val: np.ndarray | float | None = None) -> None:
        super().__init__()
        self.value = np.asarray(val) if val is not None else None

    def set(self, val: np.ndarray | float) -> None:
        self.value = np.asarray(val)

    def eval(self) -> np.ndarray:
        return self.value

    def backdiff(self) -> None:
        pass  # leaf node, gradient already accumulated by parents

    def __repr__(self) -> str:
        if self.value is None:
            return "Value(unset)"
        return f"Value(shape={self.value.shape})"


class UnaryOp(Node):
    def __init__(self, node: Node) -> None:
        super().__init__()
        self.node = node

    def backdiff(self) -> None:
        if self.node.value is None:
            raise RuntimeError("You must call .eval() before differentiation.")


class BinaryOp(Node):
    def __init__(self, node1: Node, node2: Node) -> None:
        super().__init__()
        self.node1 = node1
        self.node2 = node2

    def backdiff(self) -> None:
        if self.node1.value is None or self.node2.value is None:
            raise RuntimeError("You must call .eval() before differentiation.")


class NegOp(UnaryOp):
    def eval(self) -> np.ndarray:
        self.value = -self.node.eval()
        return self.value

    def backdiff(self) -> None:
        super().backdiff()
        self.node.grad += -self.grad

    def __repr__(self) -> str:
        shape = self.value.shape if self.value is not None else "?"
        return f"NegOp(shape={shape})"


class SumOp(UnaryOp):
    def eval(self) -> np.ndarray:
        self.value = self.node.eval().sum()
        return self.value

    def backdiff(self) -> None:
        self.node.grad += self.grad * np.ones_like(self.node.value)

    def __repr__(self) -> str:
        return (
            f"SumOp(value={self.value})"
            if self.value is not None
            else "SumOp(unevaluated)"
        )


class SquareOp(UnaryOp):
    def eval(self) -> np.ndarray:
        self.value = self.node.eval() ** 2
        return self.value

    def backdiff(self) -> None:
        super().backdiff()
        self.node.grad += 2 * self.node.value * self.grad

    def __repr__(self) -> str:
        shape = self.value.shape if self.value is not None else "?"
        return f"SquareOp(shape={shape})"


class ReLUOp(UnaryOp):
    def eval(self) -> np.ndarray:
        self.node.eval()
        self.value = self.node.value * (self.node.value > 0)
        return self.value

    def backdiff(self) -> None:
        super().backdiff()
        self.node.grad += self.grad * (self.node.value > 0)

    def __repr__(self) -> str:
        shape = self.value.shape if self.value is not None else "?"
        return f"ReLUOp(shape={shape})"


class SigmoidOp(UnaryOp):
    def eval(self) -> np.ndarray:
        self.node.eval()
        self.value = 1 / (1 + np.exp(-self.node.value))
        return self.value

    def backdiff(self) -> None:
        super().backdiff()
        self.node.grad += self.grad * self.value * (1 - self.value)

    def __repr__(self) -> str:
        shape = self.value.shape if self.value is not None else "?"
        return f"SigmoidOp(shape={shape})"


class SoftmaxOp(UnaryOp):
    def eval(self) -> np.ndarray:
        self.node.eval()
        exp = np.exp(self.node.value - self.node.value.max())
        self.value = exp / exp.sum()
        return self.value

    def backdiff(self) -> None:
        super().backdiff()
        J = np.diag(self.value.reshape(-1)) - self.value @ self.value.T
        self.node.grad += J.T @ self.grad

    def __repr__(self) -> str:
        shape = self.value.shape if self.value is not None else "?"
        return f"SoftmaxOp(shape={shape})"


class CrossEntropyOp(BinaryOp):
    def eval(self) -> np.ndarray:
        self.value = -(self.node1.eval() * np.log(self.node2.eval() + eps)).sum()
        return self.value

    def backdiff(self) -> None:
        super().backdiff()
        self.node1.grad += self.grad * (-np.log(self.node2.value + eps))
        self.node2.grad += self.grad * (-self.node1.value / (self.node2.value + eps))

    def __repr__(self) -> str:
        return (
            f"CrossEntropyOp(loss={self.value:.4f})"
            if self.value is not None
            else "CrossEntropyOp(unevaluated)"
        )


class AddOp(BinaryOp):
    def eval(self) -> np.ndarray:
        self.value = self.node1.eval() + self.node2.eval()
        return self.value

    def backdiff(self) -> None:
        super().backdiff()
        self.node1.grad += self.grad
        self.node2.grad += self.grad

    def __repr__(self) -> str:
        shape = self.value.shape if self.value is not None else "?"
        return f"AddOp(shape={shape})"


class MulOp(BinaryOp):
    def eval(self) -> np.ndarray:
        self.value = self.node1.eval() * self.node2.eval()
        return self.value

    def backdiff(self) -> None:
        super().backdiff()
        self.node1.grad += self.grad * self.node2.value
        self.node2.grad += self.grad * self.node1.value

    def __repr__(self) -> str:
        shape = self.value.shape if self.value is not None else "?"
        return f"MulOp(shape={shape})"


class MatMulOp(BinaryOp):
    def eval(self) -> np.ndarray:
        self.value = self.node1.eval() @ self.node2.eval()
        return self.value

    def backdiff(self) -> None:
        super().backdiff()
        self.node1.grad += self.grad @ self.node2.value.T
        self.node2.grad += self.node1.value.T @ self.grad

    def __repr__(self) -> str:
        if self.value is not None:
            return f"MatMulOp({self.node1.value.shape} @ {self.node2.value.shape})"
        return "MatMulOp(unevaluated)"
