# Tests for the Weather Intelligence API. Open-Meteo is never called for
# real here - build_prediction_row is monkeypatched with a fixed row of
# feature values instead.

import datetime as dt

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import app, cci_metadata, whc_metadata, pick_whc_label

client = TestClient(app)

SYDNEY_TODAY = dt.datetime.now(main.SYDNEY_TZ).date()


def fake_row():
    """A single row with every feature column both models need, using
    plausible round-number values - the exact values don't matter for these
    tests, only that predictions run and come back in the right shape."""
    all_columns = set(cci_metadata["features"]) | set(whc_metadata["features"])
    return pd.Series({col: 1.0 for col in all_columns})


@pytest.fixture(autouse=True)
def mock_weather_fetch(monkeypatch):
    """Every test gets Open-Meteo mocked out by default; individual tests
    can still monkeypatch further for error-path testing."""
    monkeypatch.setattr(main, "build_prediction_row", lambda input_date: fake_row())
    main._feature_cache.clear()


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["project"] == "Weather Intelligence API"
    assert "/health" in body["endpoints"]
    assert "comfort_climate" in body["prediction_targets"]
    assert "weather_hazard" in body["prediction_targets"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "message": "Weather Intelligence API is running"}


def test_model_metadata():
    response = client.get("/model-metadata")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    targets = {item["target"] for item in body}
    assert targets == {"Climate Comfort Index", "Weather Hazard Category"}


def test_predict_comfort_climate_success():
    response = client.get("/predict/index/comfort_climate", params={"date": "2024-06-15"})
    assert response.status_code == 200
    body = response.json()
    assert body["input_date"] == "2024-06-15"
    predictions = body["predictions"]["comfort_climate"]
    assert list(predictions.keys()) == ["2024-06-16", "2024-06-17", "2024-06-18"]
    for value in predictions.values():
        assert isinstance(value, int)
        assert 0 <= value <= 100


def test_predict_weather_hazard_success():
    response = client.get("/predict/category/weather_hazard", params={"date": "2024-06-15"})
    assert response.status_code == 200
    body = response.json()
    assert body["input_date"] == "2024-06-15"
    predictions = body["predictions"]["weather_hazard"]
    assert list(predictions.keys()) == ["2024-06-22"]
    assert list(predictions.values())[0] in ("Low Risk", "Moderate Risk", "High Risk", "Extreme Risk")


def test_pick_whc_label_high_risk():
    # high + extreme above threshold, high is the bigger of the two
    label = pick_whc_label([0.1, 0.1, 0.6, 0.2])
    assert label == "High Risk"


def test_pick_whc_label_extreme_risk():
    label = pick_whc_label([0.1, 0.1, 0.2, 0.6])
    assert label == "Extreme Risk"


def test_pick_whc_label_low_risk():
    # high + extreme below threshold, low is the bigger of the two
    label = pick_whc_label([0.6, 0.3, 0.05, 0.05])
    assert label == "Low Risk"


def test_pick_whc_label_moderate_risk():
    label = pick_whc_label([0.3, 0.6, 0.05, 0.05])
    assert label == "Moderate Risk"


def test_missing_date_returns_422():
    response = client.get("/predict/index/comfort_climate")
    assert response.status_code == 422


def test_invalid_date_format_returns_422():
    response = client.get("/predict/index/comfort_climate", params={"date": "banana"})
    assert response.status_code == 422


def test_impossible_date_returns_422():
    response = client.get("/predict/index/comfort_climate", params={"date": "2025-13-40"})
    assert response.status_code == 422


def test_future_date_returns_400():
    future_date = (SYDNEY_TODAY + dt.timedelta(days=1)).isoformat()
    response = client.get("/predict/index/comfort_climate", params={"date": future_date})
    assert response.status_code == 400


def test_today_returns_400():
    response = client.get("/predict/index/comfort_climate", params={"date": SYDNEY_TODAY.isoformat()})
    assert response.status_code == 400


def test_date_too_early_returns_400():
    response = client.get("/predict/index/comfort_climate", params={"date": "1900-01-01"})
    assert response.status_code == 400


def test_open_meteo_down_returns_503(monkeypatch):
    def raise_error(input_date):
        raise RuntimeError("Open-Meteo request failed")

    monkeypatch.setattr(main, "build_prediction_row", raise_error)
    response = client.get("/predict/index/comfort_climate", params={"date": "2024-06-15"})
    assert response.status_code == 503


def test_unknown_route_returns_404():
    response = client.get("/not-a-real-endpoint")
    assert response.status_code == 404


def fake_hourly_df(start_date, end_date):
    """Hourly weather data covering start_date to end_date, with a made-up
    but plausible value for every raw feature column."""
    hours = pd.date_range(start_date, dt.datetime.combine(end_date, dt.time(23)), freq="h")
    data = {"time": hours}
    for col in main.RAW_FEATURE_COLUMNS:
        data[col] = 10.0
    return pd.DataFrame(data)


def test_fetch_daily_window_uses_archive_when_it_covers_the_date(monkeypatch):
    end_date = dt.date(2024, 6, 15)
    start_date = end_date - dt.timedelta(days=main.LOOKBACK_DAYS - 1)

    monkeypatch.setattr(main, "fetch_hourly", lambda *a, **k: fake_hourly_df(start_date, end_date))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("forecast fallback should not be used when the archive has full coverage")

    monkeypatch.setattr(main, "fetch_forecast_hourly", fail_if_called)

    daily_df = main.fetch_daily_window(end_date)
    assert daily_df["date"].max().date() == end_date


def test_fetch_daily_window_falls_back_when_archive_lags(monkeypatch):
    end_date = dt.date(2024, 6, 15)
    start_date = end_date - dt.timedelta(days=main.LOOKBACK_DAYS - 1)
    archive_end = end_date - dt.timedelta(days=3)  # archive hasn't caught up yet

    monkeypatch.setattr(main, "fetch_hourly", lambda *a, **k: fake_hourly_df(start_date, archive_end))

    forecast_called = {"value": False}

    def fake_forecast(start, end):
        forecast_called["value"] = True
        return fake_hourly_df(start_date, end_date)

    monkeypatch.setattr(main, "fetch_forecast_hourly", fake_forecast)

    daily_df = main.fetch_daily_window(end_date)
    assert forecast_called["value"] is True
    assert daily_df["date"].max().date() == end_date
