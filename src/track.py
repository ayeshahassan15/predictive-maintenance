import numpy as np
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
X, y, groups = df[feature_cols], df["RUL"], df["unit"]

models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
    "xgboost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42),
}

mlflow.set_experiment("rul-fd001")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pred = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=5))
        mlflow.log_param("model", name)
        mlflow.log_param("window", 5)
        mlflow.log_param("rul_cap", 125)
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mean_absolute_error(y, pred))
        mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y, pred)))
        mlflow.log_metric("r2", r2_score(y, pred))
        print(name, "logged")