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


def test_report_and_health_endpoints():
    from fastapi.testclient import TestClient
    from server import app
    c = TestClient(app)
    assert c.get("/api/health").json()["status"] == "ok"
    rep = c.get("/api/report").json()
    assert "one_step" in rep and "twenty_four_hour_recursive" in rep


def test_forecast_param_clamping():
    from fastapi.testclient import TestClient
    from server import app
    c = TestClient(app)
    r = c.get("/api/forecast?history=99999&horizon=99999")
    d = r.json()
    assert len(d["actual"]) <= 240
    assert len(d["forecast"]["p50"]) <= 48
