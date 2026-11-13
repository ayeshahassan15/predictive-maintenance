import pandas as pd
from features import add_rolling_features
from data import add_rul, cap_rul, useful_sensors


def make_df():
    df = pd.DataFrame({
        "unit": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
    })
    for s in useful_sensors:
        df[s] = [10.0, 20.0, 30.0, 100.0, 200.0]
    return df


def test_add_rul_last_cycle_is_zero():
    df = add_rul(make_df())
    last = df.groupby("unit").tail(1)
    assert (last["RUL"] == 0).all()


def test_add_rul_values():
    df = add_rul(make_df())
    assert df[df["unit"] == 1]["RUL"].tolist() == [2, 1, 0]


def test_cap_rul():
    df = pd.DataFrame({"RUL": [300, 125, 50]})
    assert cap_rul(df, cap=125)["RUL"].tolist() == [125, 125, 50]


def test_rolling_does_not_mix_engines():
    df = add_rolling_features(make_df(), window=2)
    first_of_engine_2 = df[(df["unit"] == 2) & (df["cycle"] == 1)].iloc[0]
    assert first_of_engine_2["sensor_2_mean"] == 100.0
    assert first_of_engine_2["sensor_2_std"] == 0.0