from src.pv_array import PVArray
import logging
import os

logging.basicConfig(level=logging.DEBUG)


def test_output():
    PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
    PVARRAY_CKP_PATH = os.path.join("data", "01_pvarray_iv.json")
    pv_array = PVArray.from_json(PV_PARAMS_PATH, ckp_path=PVARRAY_CKP_PATH)
    result = pv_array.simulate(28.459, 1000, 25)
    assert abs(result.power - 178.59853205) < 1e-1
    assert abs(result.voltage - 28.459) < 1e-1
    assert abs(result.current - 6.27564328) < 1e-1


def test_real_mpp():
    PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
    PVARRAY_CKP_PATH = os.path.join("data", "01_pvarray_iv.json")
    pv_array = PVArray.from_json(PV_PARAMS_PATH, ckp_path=PVARRAY_CKP_PATH)
    result = pv_array.get_true_mpp([1000], [25])
    assert abs(result.power - 199.4804595) < 1e-1
    assert abs(result.voltage - 26.32) < 1e-1
    assert abs(result.current - 7.57904481) < 1e-1
