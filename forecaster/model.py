"""A small neural network written in raw NumPy, trained with pinball loss.

Architecture: input -> 64 (ReLU) -> 32 (ReLU) -> 3 outputs.
The three outputs are the 10th, 50th and 90th percentiles of next hour's
CO - a real prediction interval. Derivation of everything: docs/MATH.md.
"""
import numpy as np


def pinball_loss(pred: np.ndarray, y: np.ndarray, taus=(0.1, 0.5, 0.9)) -> float:
    """The quantile loss. Underpredict and overpredict cost differently
    unless tau = 0.5 (then it is exactly MAE / 2)."""
    e = y[:, None] - pred                    # (n, 3) errors per quantile
    t = np.array(taus)[None, :]
    return float(np.maximum(t * e, (t - 1) * e).mean())


def pinball_grad(pred: np.ndarray, y: np.ndarray, taus=(0.1, 0.5, 0.9)) -> np.ndarray:
    """d loss / d prediction. Piecewise constant - that is what makes
    quantile regression cheap and robust."""
    e = y[:, None] - pred
    t = np.array(taus)[None, :]
    return np.where(e > 0, -t, 1 - t) / pred.size


class MLP:
    def __init__(self, n_in: int, n_h1=64, n_h2=32, n_out=3, seed=42):
        rng = np.random.default_rng(seed)
        # He initialization: keeps gradients healthy through ReLU layers.
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_h1)); self.b1 = np.zeros(n_h1)
        self.W2 = rng.normal(0, np.sqrt(2 / n_h1), (n_h1, n_h2)); self.b2 = np.zeros(n_h2)
        self.W3 = rng.normal(0, np.sqrt(2 / n_h2), (n_h2, n_out)); self.b3 = np.zeros(n_out)

    @staticmethod
    def _softplus(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def forward(self, X, raw=False):
        """Returns (p10, p50, p90) columns. The net emits a median plus two
        unconstrained deltas; softplus makes the deltas positive, so the
        interval can never cross - p10 <= p50 <= p90 by construction."""
        self._X = X
        self._z1 = X @ self.W1 + self.b1
        self._a1 = np.maximum(0, self._z1)            # ReLU
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = np.maximum(0, self._z2)
        self._raw = self._a2 @ self.W3 + self.b3      # (n, 3): median, d_lo, d_hi
        if raw:
            return self._raw
        q50 = self._raw[:, 1:2]
        q10 = q50 - self._softplus(self._raw[:, 0:1])
        q90 = q50 + self._softplus(self._raw[:, 2:3])
        return np.hstack([q10, q50, q90])

    def predict(self, X):
        """Stateless inference pass - safe to call concurrently, unlike
        forward() which caches activations for backward()."""
        z1 = np.maximum(0, X @ self.W1 + self.b1)
        z2 = np.maximum(0, z1 @ self.W2 + self.b2)
        raw = z2 @ self.W3 + self.b3
        q50 = raw[:, 1:2]
        q10 = q50 - self._softplus(raw[:, 0:1])
        q90 = q50 + self._softplus(raw[:, 2:3])
        return np.hstack([q10, q50, q90])

    def backward(self, grad_out, lr):
        """Plain backprop, each line matching docs/MATH.md.

        grad_out is d loss / d (q10, q50, q90); we chain through the
        softplus reparameterization to get d loss / d raw outputs."""
        sig = lambda x: 1 / (1 + np.exp(-x))
        g_raw = np.empty_like(self._raw)
        g_raw[:, 1:2] = grad_out[:, 0:1] + grad_out[:, 1:2] + grad_out[:, 2:3]
        g_raw[:, 0:1] = -grad_out[:, 0:1] * sig(self._raw[:, 0:1])
        g_raw[:, 2:3] = grad_out[:, 2:3] * sig(self._raw[:, 2:3])
        grad_out = g_raw
        dW3 = self._a2.T @ grad_out; db3 = grad_out.sum(0)
        da2 = grad_out @ self.W3.T
        dz2 = da2 * (self._z2 > 0)                    # ReLU gate
        dW2 = self._a1.T @ dz2; db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self._z1 > 0)
        dW1 = self._X.T @ dz1; db1 = dz1.sum(0)
        for W, dW, b, db in ((self.W1, dW1, self.b1, db1),
                             (self.W2, dW2, self.b2, db2),
                             (self.W3, dW3, self.b3, db3)):
            W -= lr * dW
            b -= lr * db

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(d["W1"].shape[0], d["W1"].shape[1], d["W2"].shape[1], d["W3"].shape[1])
        m.W1, m.b1, m.W2, m.b2, m.W3, m.b3 = d["W1"], d["b1"], d["W2"], d["b2"], d["W3"], d["b3"]
        return m
