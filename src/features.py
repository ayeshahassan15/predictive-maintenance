from data import useful_sensors


def add_rolling_features(df, window=5):
    df = df.sort_values(["unit", "cycle"]).copy()
    for s in useful_sensors:
        g = df.groupby("unit")[s]
        df[f"{s}_mean"] = g.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{s}_std"] = g.transform(
            lambda x: x.rolling(window, min_periods=1).std()
        ).fillna(0)
    return df