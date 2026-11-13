import joblib
from xgboost import XGBRegressor

from data import load_train, add_rul, cap_rul, useful_sensors
from features import add_rolling_features

WINDOW = 5

df = cap_rul(add_rul(load_train("data/raw/train_FD001.txt")))
df = add_rolling_features(df, window=WINDOW)

feature_cols = (
    useful_sensors
    + [f"{s}_mean" for s in useful_sensors]
    + [f"{s}_std" for s in useful_sensors]
)

model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42)
model.fit(df[feature_cols], df["RUL"])

joblib.dump(
    {"model": model, "feature_cols": feature_cols, "window": WINDOW},
    "models/rul_model.joblib",
)
print("Saved models/rul_model.joblib")

# Check: load it back and predict
bundle = joblib.load("models/rul_model.joblib")
sample = df[bundle["feature_cols"]].head(3)
print(bundle["model"].predict(sample))