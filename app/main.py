# Weather Intelligence API - 36120 AT2 (Shruthi Murali, 26170922)
#
# Serves two models trained in the experiments repo on Sydney weather data:
# - Climate Comfort Index (CCI): regression, predicts D+1/D+2/D+3
# - Weather Hazard Category (WHC): classification, predicts D+7
#
# Both predictions reuse the same package functions used for training
# (fetch_hourly, to_daily, compute_cci, compute_whi, build_features), so the
# features seen at prediction time are built the same way as in the notebooks.

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query

from nba_career_predictor.weather import (
    CLASS_LABELS,
    build_features,
    compute_cci,
    compute_whi,
    fetch_hourly,
    to_daily,
)

# --- Location and lookback settings (must match training) -----------------

SYDNEY_TZ = ZoneInfo("Australia/Sydney")

# Both models' feature lists come from lags/rolling windows of these raw
# variables (plus the derived cci/whi columns) - see the experiments repo's
# comfort_climate/weather_hazard notebooks, Section D "Feature Engineering".
RAW_FEATURE_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m",
    "cloud_cover", "precipitation", "snowfall", "dew_point_2m", "surface_pressure",
    "shortwave_radiation", "wind_direction_10m", "et0_fao_evapotranspiration",
]

# The longest feature window either model uses is a 30-day rolling window
# plus a 30-day lag, which needs 31 days of history ending on the input
# date. 35 days gives a small safety buffer.
LOOKBACK_DAYS = 35

# Open-Meteo's historical archive starts in 1940, so a valid input date must
# leave room for the full lookback window before that.
ARCHIVE_START = dt.date(1940, 1, 1)
MIN_VALID_DATE = ARCHIVE_START + dt.timedelta(days=LOOKBACK_DAYS - 1)

FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# --- Load models and metadata once at startup ------------------------------

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

cci_models = {
    horizon: joblib.load(MODELS_DIR / "comfort_climate" / f"cci_{horizon}_model.joblib")
    for horizon in ("d1", "d2", "d3")
}
with open(MODELS_DIR / "comfort_climate" / "metadata.json") as f:
    cci_metadata = json.load(f)

whc_model = joblib.load(MODELS_DIR / "weather_hazard" / "whc_model.joblib")
with open(MODELS_DIR / "weather_hazard" / "metadata.json") as f:
    whc_metadata = json.load(f)

# predict_proba's columns follow whc_model.classes_, which sklearn always
# sorts ascending for integer labels - checked once here rather than on
# every request.
assert list(whc_model.classes_) == [0, 1, 2, 3]

WHC_THRESHOLD = whc_metadata["lift_analysis"]["logistic_regression_tuned_threshold"]["threshold"]

app = FastAPI(
    title="Weather Intelligence API",
    version="1.0.0",
    description=(
        "Predicts Sydney's Climate Comfort Index (3-day outlook) and Weather "
        "Hazard Category (7-day outlook) from Open-Meteo historical weather data."
    ),
)

# --- Fetching and feature building ------------------------------------------

# Small in-memory cache so the two prediction endpoints don't both re-fetch
# Open-Meteo data when called for the same date.
_feature_cache: dict[dt.date, pd.Series] = {}


def fetch_forecast_hourly(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Fall back to the Open-Meteo forecast API's past_days data, for recent
    dates the historical archive hasn't caught up to yet."""
    today = dt.datetime.now(SYDNEY_TZ).date()
    past_days = min((today - start_date).days, 92)
    params = {
        "latitude": -33.8688,
        "longitude": 151.2093,
        "hourly": ",".join(RAW_FEATURE_COLUMNS),
        "timezone": "Australia/Sydney",
        "past_days": past_days,
        "forecast_days": 1,
    }
    response = requests.get(FORECAST_API_URL, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Open-Meteo forecast API failed (status {response.status_code})")

    hourly_df = pd.DataFrame(response.json()["hourly"])
    hourly_df["time"] = pd.to_datetime(hourly_df["time"])
    hourly_df = hourly_df[hourly_df["time"].dt.date <= end_date]
    return hourly_df.sort_values("time").reset_index(drop=True)


def fetch_daily_window(end_date: dt.date) -> pd.DataFrame:
    """Get LOOKBACK_DAYS of daily weather data ending at end_date."""
    start_date = end_date - dt.timedelta(days=LOOKBACK_DAYS - 1)

    hourly_df = None
    try:
        hourly_df = fetch_hourly(start_date.isoformat(), end_date.isoformat(), RAW_FEATURE_COLUMNS)
    except RuntimeError:
        pass

    # The archive lags a few days behind real time, so if it failed or
    # doesn't yet cover end_date, use the forecast API instead (it also
    # returns actually-recorded weather for recent past days, not just
    # forecasts) - this is a documented limitation, see the README.
    if hourly_df is None or hourly_df["time"].dt.date.max() < end_date:
        hourly_df = fetch_forecast_hourly(start_date, end_date)

    return to_daily(hourly_df)


def build_prediction_row(input_date: dt.date) -> pd.Series:
    """Fetch weather up to input_date and build the same features used in
    training, returning the one row for input_date."""
    if input_date in _feature_cache:
        return _feature_cache[input_date]

    daily_df = fetch_daily_window(input_date)
    daily_df = compute_cci(daily_df)
    daily_df = compute_whi(daily_df)
    features_df = build_features(daily_df, feature_columns=RAW_FEATURE_COLUMNS + ["cci", "whi"])

    match = features_df[features_df["date"] == pd.Timestamp(input_date)]
    if match.empty:
        raise RuntimeError(f"Open-Meteo returned no weather data for {input_date}")

    row = match.iloc[0]
    _feature_cache[input_date] = row
    return row


def validate_date(input_date: dt.date) -> None:
    sydney_today = dt.datetime.now(SYDNEY_TZ).date()
    if input_date >= sydney_today:
        yesterday = sydney_today - dt.timedelta(days=1)
        raise HTTPException(
            status_code=400,
            detail=f"date must be yesterday or earlier (Sydney time); latest valid date is {yesterday}",
        )
    if input_date < MIN_VALID_DATE:
        raise HTTPException(
            status_code=400,
            detail=f"date is too early; the earliest valid date is {MIN_VALID_DATE}",
        )


def pick_whc_label(class_probabilities: np.ndarray) -> str:
    """Turn the model's 4 class probabilities into a label, using the
    validation-tuned threshold instead of plain argmax (see metadata.json's
    lift_analysis: the default argmax rule is no better than chance, this
    tuned threshold is the only operating point that beats it)."""
    low, moderate, high, extreme = class_probabilities
    if high + extreme >= WHC_THRESHOLD:
        predicted_class = 2 if high >= extreme else 3
    else:
        predicted_class = 0 if low >= moderate else 1
    return CLASS_LABELS[predicted_class]


# --- Endpoints ---------------------------------------------------------------

API_GITHUB_URL = "https://github.com/shruthimurali05/AT2-api"

# Shared error response docs for the two predict endpoints, so /docs shows a
# real example instead of a generic "string" schema for each status code.
COMMON_ERROR_RESPONSES = {
    400: {
        "description": "Date too early, or today/in the future (Sydney time)",
        "content": {"application/json": {"example": {"detail": "date must be yesterday or earlier (Sydney time); latest valid date is 2026-09-26"}}},
    },
    404: {
        "description": "Unknown route",
        "content": {"application/json": {"example": {"detail": "Not Found"}}},
    },
    503: {
        "description": "Open-Meteo is unreachable or timed out",
        "content": {"application/json": {"example": {"detail": "Open-Meteo weather service is unavailable: ..."}}},
    },
}


@app.get("/")
def root():
    return {
        "project": "Weather Intelligence API",
        "description": (
            "Predicts Sydney's Climate Comfort Index (3-day outlook) and Weather "
            "Hazard Category (7-day outlook) from Open-Meteo historical weather data."
        ),
        "version": "1.0.0",
        "github": API_GITHUB_URL,
        "endpoints": [
            "/",
            "/health",
            "/predict/index/comfort_climate",
            "/predict/category/weather_hazard",
            "/model-metadata",
        ],
        "prediction_targets": {
            "comfort_climate": "Climate Comfort Index, 0-100, regression, predicted for D+1, D+2 and D+3.",
            "weather_hazard": "Weather Hazard Category, one of Low Risk / Moderate Risk / High Risk / Extreme Risk, predicted for exactly D+7.",
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy", "message": "Weather Intelligence API is running"}


@app.get(
    "/predict/index/comfort_climate",
    responses={
        200: {
            "content": {"application/json": {"example": {
                "input_date": "2025-01-01",
                "predictions": {"comfort_climate": {"2025-01-02": 72, "2025-01-03": 76, "2025-01-04": 81}},
            }}},
        },
        **COMMON_ERROR_RESPONSES,
    },
)
def predict_comfort_climate(
    date: dt.date = Query(..., description="Input date (YYYY-MM-DD), Sydney time"),
):
    validate_date(date)
    try:
        row = build_prediction_row(date)
    except (requests.exceptions.RequestException, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Open-Meteo weather service is unavailable: {exc}")

    X = row[cci_metadata["features"]].to_frame().T.astype(float)

    predictions = {}
    for i, horizon in enumerate(("d1", "d2", "d3"), start=1):
        pred = cci_models[horizon].predict(X)[0]
        pred = round(float(np.clip(pred, 0, 100)))
        target_date = date + dt.timedelta(days=i)
        predictions[target_date.isoformat()] = pred

    return {"input_date": date.isoformat(), "predictions": {"comfort_climate": predictions}}


@app.get(
    "/predict/category/weather_hazard",
    responses={
        200: {
            "content": {"application/json": {"example": {
                "input_date": "2025-01-01",
                "predictions": {"weather_hazard": {"2025-01-08": "Moderate Risk"}},
            }}},
        },
        **COMMON_ERROR_RESPONSES,
    },
)
def predict_weather_hazard(
    date: dt.date = Query(..., description="Input date (YYYY-MM-DD), Sydney time"),
):
    validate_date(date)
    try:
        row = build_prediction_row(date)
    except (requests.exceptions.RequestException, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Open-Meteo weather service is unavailable: {exc}")

    X = row[whc_metadata["features"]].to_frame().T.astype(float)
    proba = whc_model.predict_proba(X)[0]
    label = pick_whc_label(proba)

    target_date = date + dt.timedelta(days=7)
    return {"input_date": date.isoformat(), "predictions": {"weather_hazard": {target_date.isoformat(): label}}}


@app.get("/model-metadata")
def model_metadata():
    return [cci_metadata, whc_metadata]
