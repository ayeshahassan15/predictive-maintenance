import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "api"))

from features import add_rolling_features  # noqa: E402
from schemas import RULRequest, RULResponse  # noqa: E402

bundle = joblib.load(ROOT / "models" / "rul_model.joblib")
model = bundle["model"]
feature_cols = bundle["feature_cols"]
window = bundle["window"]

app = FastAPI(title="Engine RUL Predictor")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-rul", response_model=RULResponse)
def predict_rul(request: RULRequest):
    df = pd.DataFrame([r.model_dump() for r in request.readings])
    df["unit"] = 1
    df = add_rolling_features(df, window=window)
    last = df.tail(1)
    pred = float(model.predict(last[feature_cols])[0])
    pred = min(max(pred, 0.0), 125.0)
    return RULResponse(predicted_rul=round(pred, 2), cycles_used=len(df))