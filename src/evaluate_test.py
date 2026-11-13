import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from data import load_train, add_rul, cap_rul, load_rul, useful_sensors
from features import add_rolling_features

feature_cols = (
    useful_sensors
    + [f"{s}_mean" for s in useful_sensors]
    + [f"{s}_std" for s in useful_sensors]
)

# 1. Train on all training data
train = cap_rul(add_rul(load_train("data/raw/train_FD001.txt")))
train = add_rolling_features(train, window=5)

model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42)
model.fit(train[feature_cols], train["RUL"])

# 2. Build features on test data
test = load_train("data/raw/test_FD001.txt")
test = add_rolling_features(test, window=5)

# 3. Keep only the last cycle of each engine
last = test.groupby("unit").tail(1)

# 4. Compare with the true RUL
true_rul = load_rul("data/raw/RUL_FD001.txt")["RUL"].clip(upper=125).values
pred = model.predict(last[feature_cols])

print("Test engines:", len(last))
print("MAE :", round(mean_absolute_error(true_rul, pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(true_rul, pred)), 2))
print("R2  :", round(r2_score(true_rul, pred), 3))