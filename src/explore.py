# import matplotlib.pyplot as plt
# from data import load_train, add_rul, useful_sensors, cap_rul


# #import matplotlib.pyplot as plt: loads the plotting tools with the short name plt. engine = 
# # df[df["unit"] == 1]: keeps only the rows of engine 1. We look at one engine first, so the picture is clear. 
# # plt.subplots(3, 1, ...): makes 3 small charts stacked vertically. figsize is the size. sharex=True 
# # means all 3 use the same cycle axis.
# #  for ax, s in zip(axes, [...]): goes through each chart and its sensor name together. ax.plot(x, y): draws 
# # the line. Here x is the cycle and y is the sensor value. ax.set_ylabel(s): writes the sensor name beside each 
# # chart. plt.show(): opens the chart window. Close the window to get your terminal back. 


# # df = add_rul(load_train("data/raw/train_FD001.txt"))
# # engine = df[df["unit"] == 1]

# # fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
# # for ax, s in zip(axes, ["sensor_2", "sensor_4", "sensor_11"]):
# #     ax.plot(engine["cycle"], engine[s])
# #     ax.set_ylabel(s)
# # axes[-1].set_xlabel("cycle")
# # plt.show()

# # print(df[sensors].std().sort_values())


# # clip(upper=125) replaces every value above 125 with 125. Smaller values stay the same.
# # cap=125 is a default value. You can change it later without rewriting the function.
# # .iloc[[0, 50, 100, 150, 191]] picks rows by position, so we see 5 rows spread across engine 1’s life.
# # Capping means setting a maximum value. Any number above the maximum is replaced by the maximum.
# df = add_rul(load_train("data/raw/train_FD001.txt"))
# print("Before cap, max RUL:", df["RUL"].max())

# df = cap_rul(df)
# print("After cap, max RUL:", df["RUL"].max())
# print(df[df["unit"] == 1][["cycle", "RUL"]].iloc[[0, 50, 100, 150, 191]])


# # print(len(useful_sensors))
# # print(useful_sensors)


from data import load_train, add_rul, cap_rul
from features import add_rolling_features

df = cap_rul(add_rul(load_train("data/raw/train_FD001.txt")))
df = add_rolling_features(df, window=5)

print(df.shape)
cols = ["unit", "cycle", "sensor_2", "sensor_2_mean", "sensor_2_std"]
print(df[cols].head(6))
print(df[df["unit"] == 2][cols].head(2))