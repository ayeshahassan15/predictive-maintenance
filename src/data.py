import pandas as pd

sensors = [f"sensor_{i}" for i in range(1, 22)]
columns = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + sensors


def load_train(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    return df



def add_rul(df):
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    return df

drop_sensors = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]
useful_sensors = [s for s in sensors if s not in drop_sensors]


def cap_rul(df, cap=125):
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def load_rul(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])



if __name__ == "__main__":
    df = load_train("data/raw/train_FD001.txt")
    print(df.head())
    print(df.shape)
    
    life = df.groupby("unit")["cycle"].max()
    print("Number of engines:", df["unit"].nunique())
    print("Shortest life:", life.min())
    print("Longest life:", life.max())
    print("Average life:", life.mean())

    df = add_rul(df)
    print(df[["unit", "cycle", "RUL"]].head())
    print(df[["unit", "cycle", "RUL"]].tail())