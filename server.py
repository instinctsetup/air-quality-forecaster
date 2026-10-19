"""air-quality-forecaster web server.

Run:  python train.py && uvicorn server:app --port 8000
Then open http://localhost:8000
"""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from forecaster import data, forecast, model

app = FastAPI(title="air-quality-forecaster")
FRONTEND = Path(__file__).parent / "frontend"
ROWS = data.load_series()
MODEL = model.MLP.load(Path(__file__).parent / "forecaster" / "quantile_mlp.npz")
S = np.load(Path(__file__).parent / "forecaster" / "scalers.npz")
LAGS = 24


@app.get("/api/health")
def health():
    return {"status": "ok", "hours_of_history": len(ROWS)}


@app.get("/api/report")
def report():
    return json.loads((Path(__file__).parent / "docs" / "eval-report.json").read_text())


@app.get("/api/forecast")
def get_forecast(history: int = 72, horizon: int = 24):
    """Actual CO for the last `history` hours + a recursive `horizon`-hour
    forecast with p10/p50/p90 fan."""
    history = max(LAGS, min(history, 240))
    horizon = max(1, min(horizon, 48))
    tail = ROWS[-history:]
    actual = [{"date": r["date"], "time": r["time"], "co": r["co"]} for r in tail]
    seed = [r["co"] for r in ROWS[-LAGS:]]
    lo, mid, hi = forecast.recursive_forecast(
        MODEL, seed, len(ROWS), horizon, S["x_mu"], S["x_sd"], S["y_mu"], S["y_sd"], lags=LAGS)
    return {"actual": actual, "forecast": {"p10": lo, "p50": mid, "p90": hi},
            "note": "Recursive multi-step forecast: each prediction is fed back as input."}


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")

app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
