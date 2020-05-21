import pandas as pd
import os
import json


def parse_depfie_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")
    df = df.set_index("Date")
    return df


def save_dict_to_file(dic, path):
    with open(path, "w") as f:
        json.dump(dic, f)


def load_dict_from_file(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    df = parse_depfie_csv(os.path.join(".", "data", "weather.csv"))[79:]

    print(df.head(), end="\n\n")
    print(df.tail(), end="\n\n")
    print(df.loc["2019-06-29 07:20:00"])
    print(df.iloc[0])
