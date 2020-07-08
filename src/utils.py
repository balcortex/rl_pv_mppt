import os

import numpy as np
import pandas as pd


def read_weather_csv(path: str) -> pd.DataFrame:
    "Read a csv file and returns a DataFrame object"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")
    df = df.set_index("Date")
    return df


def clip_num(value: float, minimum: float = -np.inf, maximum: float = np.inf):
    "Clip the value between minimum and maximum parameters"
    return min(max(value, minimum), maximum)

