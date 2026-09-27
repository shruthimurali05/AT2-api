# AT2 – Weather Intelligence API

**Student:** Shruthi Murali
**Student ID:** 26170922
**Subject:** 36120 Advanced Machine Learning Application – Spring 2026, AT2

FastAPI application serving two machine learning models trained on Open-Meteo
historical weather data for Sydney. Deployed on Render.

- Live API: `<add Render URL>` (Phase 6)
- Experimentation repository: https://github.com/shruthimurali05/AT2---Machine-Learning-as-a-Service
- Package repository: https://github.com/shruthimurali05/36120-26SP-group31-26170922-package

## Prediction targets

- **Climate Comfort Index (CCI)** – regression, 0-100, predicted for the next 3 days
  (D+1, D+2, D+3). Final model: XGBoost, one model per horizon.
- **Weather Hazard Category (WHC)** – classification into Low Risk / Moderate Risk /
  High Risk / Extreme Risk, predicted for exactly 7 days ahead (D+7). Final model:
  Logistic Regression.

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Project overview and API documentation |
| GET | `/health` | Health check |
| GET | `/predict/index/comfort_climate?date=YYYY-MM-DD` | Climate Comfort Index for the next 3 days |
| GET | `/predict/category/weather_hazard?date=YYYY-MM-DD` | Weather Hazard Category for date + 7 days |
| GET | `/model-metadata` | Information about both trained models |

`date` is the input date (Sydney time). It must be yesterday or earlier, and there
must be enough historical weather data before it (see "Valid dates" below).

### Example: comfort_climate

```
GET /predict/index/comfort_climate?date=2024-06-15
```
```json
{
  "input_date": "2024-06-15",
  "predictions": {
    "comfort_climate": {
      "2024-06-16": 54,
      "2024-06-17": 58,
      "2024-06-18": 61
    }
  }
}
```

### Example: weather_hazard

```
GET /predict/category/weather_hazard?date=2024-06-15
```
```json
{
  "input_date": "2024-06-15",
  "predictions": {
    "weather_hazard": {
      "2024-06-22": "Extreme Risk"
    }
  }
}
```

The Weather Hazard Category prediction uses a decision threshold tuned on the
validation set, not the model's plain highest-probability class. The experiments
repo found that the model's default classification catches High/Extreme Risk days
at close to a chance rate (lift below 1.0 - see `models/weather_hazard/metadata.json`,
`lift_analysis`). Thresholding `P(High Risk) + P(Extreme Risk)` against a value tuned
on the validation set (0.5347, targeting a realistic ~12% alert rate) is the only
version of this model that is genuinely better than chance (lift 1.674 on the test
set). The API applies that threshold: if the combined probability is at or above it,
it returns whichever of High Risk / Extreme Risk is more likely; otherwise it returns
whichever of Low Risk / Moderate Risk is more likely. The response is always one of
the 4 official WHC labels.

### Error responses

All errors return a JSON body with a `detail` message, never a raw traceback.

| Situation | Status |
|---|---|
| `date` missing | 422 |
| `date` badly formatted or an impossible calendar date | 422 |
| `date` too early (not enough history before it) | 400 |
| `date` is today or in the future (Sydney time) | 400 |
| Open-Meteo is unreachable or times out | 503 |
| Unknown route | 404 |

## Valid dates

A prediction needs about 35 days of daily weather history ending on `date`, so the
earliest valid date is a little after Open-Meteo's historical archive starts
(1940-01-01). The latest valid date is yesterday in Sydney time, since day D must be
a complete day of observed weather.

## Data source and limitation

Weather data comes from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
(archive-api.open-meteo.com), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The archive typically lags a few days behind real time, so for recent dates this API
falls back to Open-Meteo's forecast API (`past_days` parameter), which reports
actually-recorded weather for recent days from a different underlying model. This
means predictions for very recent dates may use slightly different input data than
predictions for older dates - noted here and in the final report as a limitation.

## Project structure

```
├── app
│   └── main.py        <- FastAPI application
├── models             <- Trained model files + metadata.json, copied from the
│                          experiments repo's final models
├── tests               <- pytest + TestClient tests, Open-Meteo mocked
├── Dockerfile          (Phase 6)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Run locally

```
conda activate at2
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs to try the endpoints interactively.

## Run tests

```
pytest tests/ -v
```

Open-Meteo is mocked in every test - no real network calls are made.

## Deploy on Render

_To be completed in Phase 6._
