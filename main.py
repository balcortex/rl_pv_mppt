from src.pv_array import PVArray
import os

PV_PARAMS_PATH = os.path.join("parameters", "pvarray_01.json")

if __name__ == "__main__":
    pvarray = PVArray.from_json(PV_PARAMS_PATH)