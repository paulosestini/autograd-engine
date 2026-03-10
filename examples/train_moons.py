"""Train a 3-layer MLP on sklearn's make_moons dataset and plot the decision boundary."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from autograd import Value


class MLP:
    def __init__(self):
        self.w1 = Value(np.random.normal(size=(16, 2), scale=1 / 2))
        self.b1 = Value(np.zeros((16, 1)))
        self.w2 = Value(np.random.normal(size=(8, 16), scale=1 / 16))
        self.b2 = Value(np.zeros((8, 1)))
        self.w3 = Value(np.random.normal(size=(2, 8), scale=1 / 8))
        self.b3 = Value(np.zeros((2, 1)))

    def forward(self, x_val):
        x = Value(x_val)
        l1 = (self.w1 @ x + self.b1).relu()
        l2 = (self.w2 @ l1 + self.b2).relu()
        l3 = self.w3 @ l2 + self.b3
        return l3.softmax()

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]


if __name__ == "__main__":
    np.random.seed(42)
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

    model = MLP()
    epochs = 50
    lr = 1e-2

    for epoch in range(epochs):
        epoch_loss = 0.0
        for xi, yi in zip(X, y):
            for p in model.parameters():
                p.zero_grad()

            y_true_arr = np.zeros((2, 1))
            y_true_arr[yi, 0] = 1.0
            y_true = Value(y_true_arr)

            z = model.forward(xi.reshape(-1, 1))
            loss = y_true.cross_entropy(z)

            epoch_loss += loss.eval()
            loss.backward()

            for p in model.parameters():
                p.value -= lr * p.grad

        print(f"epoch {epoch + 1:3d}/{epochs}  loss={epoch_loss / len(X):.4f}")

    # Decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    preds = np.zeros(xx.size)
    for i, (px, py) in enumerate(zip(xx.ravel(), yy.ravel())):
        preds[i] = np.argmax(model.forward(np.array([[px], [py]])).eval())
    preds = preds.reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, preds, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", linewidths=0.5)
    plt.title("Decision boundary, MLP trained with autograd-engine")
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "..", "decision_boundary.png"), dpi=150
    )
    print("Saved decision_boundary.png")
    plt.show()
