from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import load_train, add_rul, cap_rul, useful_sensors
from features import add_rolling_features

df = cap_rul(add_rul(load_train("data/raw/train_FD001.txt")))
df = add_rolling_features(df, window=5)

feature_cols = (
    useful_sensors
    + [f"{s}_mean" for s in useful_sensors]
    + [f"{s}_std" for s in useful_sensors]
)
X = df[feature_cols]
y = df["RUL"]
groups = df["unit"]

gkf = GroupKFold(n_splits=5)
maes, rmses, r2s = [], [], []

for train_idx, val_idx in gkf.split(X, y, groups):
    # model = LinearRegression()
    # model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
   
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[val_idx])
    maes.append(mean_absolute_error(y.iloc[val_idx], pred))
    rmses.append(np.sqrt(mean_squared_error(y.iloc[val_idx], pred)))
    r2s.append(r2_score(y.iloc[val_idx], pred))

print("MAE :", round(np.mean(maes), 2))
print("RMSE:", round(np.mean(rmses), 2))
print("R2  :", round(np.mean(r2s), 3))