"""Forecast metrics from scratch."""
import numpy as np


def mae(y, pred) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def mape(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    mask = y > 0.05          # division by ~0 poisons MAPE; standard practice
    return float(np.mean(np.abs((y[mask] - pred[mask]) / y[mask])) * 100)


def interval_coverage(y, lo, hi) -> float:
    """Fraction of true values inside [p10, p90]. A well-calibrated 80%
    interval should land near 0.80."""
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    return float(np.mean((y >= lo) & (y <= hi)))
