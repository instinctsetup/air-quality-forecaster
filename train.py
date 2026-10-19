"""Train the quantile forecaster and write artifacts + an honest eval report.

    python train.py
"""
import json
import numpy as np
from forecaster import baselines, data, forecast, metrics, model

LAGS = 24
rows = data.load_series()
X, y = data.make_windows(rows, lags=LAGS)
Xtr, ytr, Xte, yte = data.time_split(X, y)
x_mu, x_sd = data.scaler_fit(Xtr)
y_mu, y_sd = float(ytr.mean()), float(ytr.std())
Xtr_s, Xte_s = (Xtr - x_mu) / x_sd, (Xte - x_mu) / x_sd

m = model.MLP(X.shape[1])
for epoch in range(40):
    lr = 0.01 * (0.95 ** epoch)
    idx = np.random.default_rng(epoch).permutation(len(Xtr_s))
    for i in range(0, len(idx), 256):
        b = idx[i:i + 256]
        m.backward(model.pinball_grad(m.forward(Xtr_s[b]), (ytr[b] - y_mu) / y_sd), lr=lr)

m.save("forecaster/quantile_mlp.npz")
np.savez("forecaster/scalers.npz", x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd)

# ---- 1-step evaluation on the held-out last 20%
pred = m.forward(Xte_s) * y_sd + y_mu
p10, p50, p90 = pred[:, 0], pred[:, 1], pred[:, 2]
report = {
    "one_step": {
        "model_mae": round(metrics.mae(yte, p50), 3),
        "model_rmse": round(metrics.rmse(yte, p50), 3),
        "naive_mae": round(metrics.mae(yte, baselines.naive(Xte, yte)), 3),
        "seasonal_naive_mae": round(metrics.mae(yte, baselines.seasonal_naive(Xte, yte)), 3),
        "moving_average_mae": round(metrics.mae(yte, baselines.moving_average(Xte, yte)), 3),
        "interval_coverage_80pct_target": round(metrics.interval_coverage(yte, p10, p90), 3),
    }
}

# ---- 24-hour recursive evaluation over the last month of the series
em, en, es = [], [], []
start = len(rows) - 24 * 30
for t0 in range(start, len(rows) - 2 * LAGS, LAGS):
    seed = [r["co"] for r in rows[t0:t0 + LAGS]]
    actual = [rows[t0 + LAGS + h]["co"] for h in range(LAGS)]
    _, mid, _ = forecast.recursive_forecast(m, seed, t0 + LAGS, LAGS, x_mu, x_sd, y_mu, y_sd)
    em.append(metrics.mae(actual, mid))
    en.append(metrics.mae(actual, [seed[-1]] * LAGS))
    es.append(metrics.mae(actual, seed))
report["twenty_four_hour_recursive"] = {
    "model_mae": round(float(np.mean(em)), 3),
    "naive_flat_mae": round(float(np.mean(en)), 3),
    "seasonal_naive_mae": round(float(np.mean(es)), 3),
}
json.dump(report, open("docs/eval-report.json", "w"), indent=2)
print(json.dumps(report, indent=2))
