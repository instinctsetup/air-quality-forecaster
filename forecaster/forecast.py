"""Recursive multi-step forecasting: feed each prediction back as input.

This is where prediction intervals really earn their keep - uncertainty
compounds with every step, and the fan chart shows it.
"""
import math
import numpy as np


def recursive_forecast(model, seed_co: list[float], start_hour: int, horizon: int,
                       x_mu, x_sd, y_mu, y_sd, lags: int = 24):
    """Roll the model forward `horizon` hours from `seed_co` (last lags values).

    Returns (p10, p50, p90) lists of length `horizon`.
    """
    buf = list(seed_co)
    lo, mid, hi = [], [], []
    for h in range(horizon):
        hour = start_hour + h
        feats = np.array(buf[-lags:]
                         + [math.sin(2 * math.pi * (hour % 24) / 24),
                            math.cos(2 * math.pi * (hour % 24) / 24)])
        q = model.forward(((feats - x_mu) / x_sd)[None, :])[0] * y_sd + y_mu
        lo.append(float(q[0])); mid.append(float(q[1])); hi.append(float(q[2]))
        buf.append(float(q[1]))          # feed the median back
    return lo, mid, hi
