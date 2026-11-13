import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_predict
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
X = df[feature_cols]
y = df["RUL"]
groups = df["unit"]

model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42)
pred = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=5))

df["pred"] = pred
df["error"] = (df["pred"] - df["RUL"]).abs()

print("MAE when true RUL > 100:", round(df[df["RUL"] > 100]["error"].mean(), 2))
print("MAE when true RUL 50-100:", round(df[(df["RUL"] > 50) & (df["RUL"] <= 100)]["error"].mean(), 2))
print("MAE when true RUL <= 50:", round(df[df["RUL"] <= 50]["error"].mean(), 2))

plt.scatter(df["RUL"], df["pred"], s=3, alpha=0.3)
plt.plot([0, 125], [0, 125], color="red")
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.show()