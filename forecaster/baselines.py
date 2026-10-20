"""Honest baselines. A fancy model that cannot beat these is not worth
shipping - every forecaster must be compared against them."""
import numpy as np

def naive(X, y, lags=24):
    """'Tomorrow equals today': predict the most recent value."""
    return X[:, lags - 1]

def seasonal_naive(X, y, lags=24):
    """'Same hour yesterday': with 24 lags, that is the oldest lag."""
    return X[:, 0]

def moving_average(X, y, lags=24):
    """Mean of the last 24 hours."""
    return X[:, :lags].mean(1)

