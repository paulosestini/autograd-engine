"""Basic usage: define a computation, run forward pass, call .backward()."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from autograd import Value

# Build a small computation graph: softmax of a matrix-vector product
x = Value(np.array([[1.0], [2.0]]))
w = Value(np.array([[1.0, 2.0], [3.0, 4.0]]))

y = (w @ x).softmax()
z = y.sum()

# Forward pass
print("z =", z.eval())
print("y =", y.value)

# Backward pass, gradients flow back through the graph
z.backward()
print("dL/dw =\n", w.grad)
print("dL/dx =\n", x.grad)

# Zero gradients for reuse
z.zero_grad()
print("after zero_grad: w.grad =", w.grad, "  x.grad =", x.grad)
