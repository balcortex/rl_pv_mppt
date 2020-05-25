from pvmppt.pv_array import PVArray

pv_array_params = {
    "Npar": "1",
    "Nser": "1",
    "Ncell": "54",
    "Voc": "32.9",
    "Isc": "8.21",
    "Vm": "26.3",
    "Im": "7.61",
    "beta_Voc_pc": "-0.1230",
    "alpha_Isc_pc": "0.0032",
    "BAL": "on",
    "Tc": "1e-6",
}

pv_array = PVArray(pv_array_params, float_precision=8)


def test_output():
    result = pv_array.simulate(28.459, 1000, 25)
    assert abs(result.power - 178.59853205) < 1e-6
    assert abs(result.voltage - 28.459) < 1e-6
    assert abs(result.current - 6.27564328) < 1e-6


def test_real_mpp():
    result = pv_array.get_true_mpp([1000], [25])
    assert abs(result.power - 199.4804595) < 1e-6
    assert abs(result.voltage - 26.32) < 1e-6
    assert abs(result.current - 7.57904481) < 1e-6
