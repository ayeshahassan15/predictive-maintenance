import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from data import load_train, add_rul, cap_rul, useful_sensors
from features import add_rolling_features

df = cap_rul(add_rul(load_train("data/raw/train_FD001.txt")))
df = add_rolling_features(df, window=5)

feature_cols = (
    useful_sensors
    + [f"{s}_mean" for s in useful_sensors]
    + [f"{s}_std" for s in useful_sensors]
)
X, y = df[feature_cols], df["RUL"]

model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42)
model.fit(X, y)

sample = X.sample(2000, random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

shap.summary_plot(shap_values, sample, max_display=15)