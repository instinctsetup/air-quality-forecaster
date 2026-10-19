import pytest
from pathlib import Path

pytestmark = pytest.mark.skipif(
    not Path("forecaster/quantile_mlp.npz").exists(), reason="run python train.py first")


def test_forecast_endpoint():
    from fastapi.testclient import TestClient
    from server import app
    c = TestClient(app)
    r = c.get("/api/forecast?history=72&horizon=24")
    data = r.json()
    assert r.status_code == 200
    assert len(data["actual"]) == 72
    assert len(data["forecast"]["p50"]) == 24
    lo, mid, hi = data["forecast"]["p10"], data["forecast"]["p50"], data["forecast"]["p90"]
    assert all(l <= m <= h for l, m, h in zip(lo, mid, hi))
