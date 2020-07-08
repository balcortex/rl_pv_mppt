import os
from src.utils import read_weather_csv, clip_num


def test_read_weather():
    df = read_weather_csv(os.path.join(".", "data", "toy_weather.csv"))[79:]
    ir = df.loc["2019-06-29 07:20:00"]["Irradiance"]
    temp = df.iloc[0]["Temperature"]

    assert ir == 200
    assert temp == 20


def test_clip_num():
    assert clip_num(0) == 0
    assert clip_num(-1.0) == -1
    assert clip_num(1) == 1
    assert clip_num(-3, minimum=0) == 0
    assert clip_num(12, maximum=10) == 10
    assert clip_num(8, 0, 10) == 8
