"""Correctness tests: loss, metrics, baselines, model, recursion."""
import numpy as np
from forecaster import baselines, data, forecast, metrics, model


def test_pinball_loss_median_is_half_mae():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.zeros((3, 3))          # all-zero predictions
    loss = model.pinball_loss(pred, y, taus=(0.5,))
    assert abs(loss - metrics.mae(y, np.zeros(3)) / 2) < 1e-9


def test_pinball_grad_sign():
    # Underprediction (y > q) should push the prediction UP (negative grad).
    g = model.pinball_grad(np.zeros((1, 3)), np.array([5.0]))
    assert (g < 0).all()


def test_metrics_perfect():
    assert metrics.mae([1, 2], [1, 2]) == 0.0
    assert metrics.rmse([1, 2], [1, 2]) == 0.0
    assert metrics.interval_coverage([1, 2], [0, 1], [2, 3]) == 1.0


def test_baselines_read_the_right_lags():
    X = np.arange(26, dtype=float).reshape(1, 26)   # 24 lags + sin/cos
    y = np.array([99.0])
    assert baselines.naive(X, y)[0] == 23.0          # most recent
    assert baselines.seasonal_naive(X, y)[0] == 0.0  # 24 hours back


def test_model_output_is_ordered():
    m = model.MLP(26)
    pred = m.forward(np.random.default_rng(0).normal(size=(16, 26)))
    assert np.all(pred[:, 0] <= pred[:, 1]) and np.all(pred[:, 1] <= pred[:, 2])


def test_training_reduces_pinball():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 26))
    y = X[:, 23] * 0.8 + rng.normal(0, 0.1, 200)     # learnable pattern
    m = model.MLP(26)
    before = model.pinball_loss(m.forward(X), y)
    for _ in range(5):
        m.backward(model.pinball_grad(m.forward(X), y), lr=0.005)
    assert model.pinball_loss(m.forward(X), y) < before


def test_recursive_forecast_shape():
    m = model.MLP(26)
    lo, mid, hi = forecast.recursive_forecast(
        m, [1.0] * 24, 100, 12, np.zeros(26), np.ones(26), 0.0, 1.0)
    assert len(lo) == len(mid) == len(hi) == 12
    assert all(l <= md <= h for l, md, h in zip(lo, mid, hi))


def test_windows_and_split():
    rows = data.load_series()
    X, y = data.make_windows(rows)
    assert len(X) == len(y) == len(rows) - 24
    Xtr, ytr, Xte, yte = data.time_split(X, y)
    assert len(Xtr) > len(Xte) > 0
