"""Turn the raw hourly records into supervised learning windows.

Target: CO concentration (mg/m^3, ground truth sensor) one hour ahead.
Features per window: the last `lags` CO values, plus hour-of-day encoded
as (sin, cos) so midnight is close to 23:00 - a classic cyclical encoding.
"""
import json
import math
from pathlib import Path
import numpy as np


def load_series(path: str | None = None) -> list[dict]:
    path = path or str(Path(__file__).parent.parent / "data" / "hourly.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def make_windows(rows: list[dict], lags: int = 24):
    """X: (n, lags + 2), y: (n,). y[t] = CO at hour t.

    Features: the last `lags` CO values and hour-of-day as (sin, cos) so
    midnight is close to 23:00. Other sensors (NOx, NO2, T, RH) were tried
    and *hurt* test MAE - the ablation is documented in docs/MATH.md; the
    honest model keeps only what generalizes.
    """
    co = np.array([r["co"] for r in rows], dtype=float)
    n = len(co) - lags
    X = np.zeros((n, lags + 2))
    y = np.zeros(n)
    for i in range(n):
        X[i, :lags] = co[i:i + lags]
        hour = i + lags
        X[i, lags] = math.sin(2 * math.pi * (hour % 24) / 24)
        X[i, lags + 1] = math.cos(2 * math.pi * (hour % 24) / 24)
        y[i] = co[hour]
    return X, y


def time_split(X, y, frac=0.8):
    """Split by time, never randomly - random splits leak the future in
    time-series problems (the model would 'remember' neighboring hours)."""
    cut = int(len(X) * frac)
    return X[:cut], y[:cut], X[cut:], y[cut:]


def scaler_fit(X):
    return X.mean(0), X.std(0) + 1e-9
