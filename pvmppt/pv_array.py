import os
import sys
from collections import namedtuple
from functools import lru_cache, partial
from typing import Dict, List

import matlab.engine
import numpy as np
from scipy.optimize import minimize

from pvmppt.matlab_api import set_parameters

PVSimResult = namedtuple("PVSimResult", ["power", "voltage", "current"])


class PVArray:
    def __init__(self, model_params: Dict, float_precision: int = 3) -> None:
        print("Starting MATLAB engine . . .")
        self.model_params = model_params
        self.voc = float(model_params["Voc"])
        self.eng = matlab.engine.start_matlab()

        self._model_path = os.path.join("pvmppt", "matlab_model")
        self._model_name = os.path.basename(self._model_path)
        self._float_precision = float_precision
        self._running = False

        self._init()

    def step(
        self, load_voltage: float, irradiance: float, cell_temp: float
    ) -> PVSimResult:
        return self._step_cached(
            round(load_voltage, self._float_precision),
            round(irradiance, self._float_precision),
            round(cell_temp, self._float_precision),
        )

    @lru_cache(maxsize=None)
    def _step_cached(self, v: float, g: float, t: float) -> PVSimResult:
        if not self._running:
            self._prepare_for_step()

        self._set_irradiance(g)
        self._set_cell_temp(t)
        self._set_load_voltage(v)

        set_parameters(
            self.eng, self._model_name, {"SimulationCommand": ["continue", "pause"]}
        )

        power = self.eng.eval("P(end);", nargout=1)
        voltage = self.eng.eval("V(end);", nargout=1)
        current = self.eng.eval("I(end);", nargout=1)

        return PVSimResult(
            round(power, self._float_precision),
            round(voltage, self._float_precision),
            round(current, self._float_precision),
        )

    def simulate(
        self, load_voltage: float, irradiance: float, cell_temp: float
    ) -> PVSimResult:
        return self._simulate_cached(
            round(load_voltage, self._float_precision),
            round(irradiance, self._float_precision),
            round(cell_temp, self._float_precision),
        )

    @lru_cache(maxsize=None)
    def _simulate_cached(self, v: float, g: float, t: float) -> PVSimResult:
        if self._running:
            self._prepare_for_sim()

        self._set_irradiance(g)
        self._set_cell_temp(t)
        self._set_load_voltage(v)

        set_parameters(self.eng, self._model_name, {"SimulationCommand": "start"})

        self.eng.eval("pause(0.1)", nargout=0)

        # print(self.eng.workspace)
        power = self.eng.eval("P(end);", nargout=1)
        voltage = self.eng.eval("V(end);", nargout=1)
        current = self.eng.eval("I(end);", nargout=1)

        return PVSimResult(
            round(power, self._float_precision),
            round(voltage, self._float_precision),
            round(current, self._float_precision),
        )

    def get_true_mpp(
        self, irradiance: List[float], temperature: List[float], ftol: float = 1e-06
    ) -> PVSimResult:
        float_precision = self._float_precision
        self._float_precision = 8
        voltages = []
        powers = []
        currents = []

        for (idx), (g, t) in enumerate(zip(irradiance, temperature)):
            print(f"Idx: {idx}/{len(irradiance) - 1}", end=", ")

            result = self._get_true_mpp_cached(
                round(g, float_precision), round(t, float_precision), ftol
            )

            voltages.append(round(result.voltage, float_precision))
            powers.append(round(result.power, float_precision))
            currents.append(round(result.current, float_precision))
            print(f"Power = {powers[-1]}")

        self._float_precision = float_precision

        if len(powers) == 1:
            return PVSimResult(powers[0], voltages[0], currents[0])
        return PVSimResult(powers, voltages, currents)

    @lru_cache(maxsize=None)
    def _get_true_mpp_cached(self, g: float, t: float, ftol: float) -> PVSimResult:
        f_output = partial(self._get_negative_output_power, irradiance=g, cell_temp=t)
        optim_result = minimize(
            f_output, 0.8 * self.voc, method="SLSQP", options={"ftol": ftol}
        )
        assert optim_result.success == True

        v = optim_result.x[0]
        p = optim_result.fun * -1
        i = p / v

        return PVSimResult(p, v, i)

    def mppt_po(
        self,
        irradiance: List[float],
        temperature: List[float],
        v0: float = 0.0,
        v_eps: float = 0.1,
    ) -> PVSimResult:
        voltages = []
        currents = []
        powers = []
        p0 = 0
        v = 0

        for (idx), (g, t) in enumerate(zip(irradiance, temperature)):
            print(f"idx: {idx}/{len(irradiance) - 1}", end=", ")

            if idx == 0:
                sim_result = self.simulate(v0, g, t)
                i = sim_result.current
                v0 = sim_result.voltage
                p0 = sim_result.power

                v = v0
                p = p0

            else:
                sim_result = self.simulate(v, g, t)
                i = sim_result.current
                v = sim_result.voltage
                p = sim_result.power

                delta_p = p - p0
                delta_v = v - v0
                v0 = v
                p0 = p

                if not delta_p == 0:
                    if delta_p > 0:
                        if delta_v > 0:
                            v += v_eps
                        else:
                            v -= v_eps
                    else:
                        if delta_v > 0:
                            v -= v_eps
                        else:
                            v += v_eps

            voltages.append(round(v0, self._float_precision))
            currents.append(round(i, self._float_precision))
            powers.append(round(p, self._float_precision))
            print(f"Power = {powers[-1]}")

        return PVSimResult(powers, voltages, currents)

    def _init(self) -> None:
        self.eng.eval("beep off", nargout=0)
        print('model = "{}";'.format(self._model_path))
        self.eng.eval('model = "{}";'.format(self._model_path), nargout=0)
        self.eng.eval("load_system(model)", nargout=0)
        status = self.eng.eval(
            f"get_param('{self._model_name}', 'SimulationStatus')", nargout=1
        )
        assert status == "stopped"
        print(f"Status = {status}")

        set_parameters(self.eng, [self._model_name, "PV Array"], self.model_params)
        print("Model loaded succesfully.")

    def _prepare_for_step(self) -> None:
        set_parameters(
            self.eng, self._model_name, {"SimulationCommand": ["start", "pause"]}
        )
        set_parameters(self.eng, self._model_name, {"StopTime": "inf"})
        self._running = True

    def _prepare_for_sim(self) -> None:
        set_parameters(self.eng, self._model_name, {"SimulationCommand": "stop"})
        set_parameters(self.eng, self._model_name, {"StopTime": "1e-3"})
        self._running = False

    def _quit_engine(self) -> None:
        # print("Quitting engine . . .")
        self.eng.quit()

    def _get_negative_output_power(
        self, load_voltage: List[float], irradiance: float, cell_temp: float
    ) -> float:
        sim_result = self.simulate(load_voltage[0], irradiance, cell_temp)
        return -sim_result.power

    def _set_load_voltage(self, voltage: float) -> None:
        set_parameters(
            self.eng,
            [self._model_name, "Variable DC Source", "Load Voltage"],
            {"Value": str(voltage)},
        )

    def _set_irradiance(self, irradiance: float) -> None:
        set_parameters(
            self.eng, [self._model_name, "Irradiance"], {"Value": str(irradiance)}
        )

    def _set_cell_temp(self, cell_temp: float) -> None:
        set_parameters(
            self.eng, [self._model_name, "Cell Temperature"], {"Value": str(cell_temp)}
        )


if __name__ == "__main__":
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

    result = pv_array.simulate(28.459, 1000, 25)
    print("Simulation")
    print(f"Inputs: V={28.459}, G={1000}, T={25}")
    print(f"Output: V={result.voltage}, I={result.current}, P={result.power}")

    result = pv_array.get_true_mpp([1000], [25])
    print("MPP Simulation")
    print(f"Inputs: G={1000}, T={25}")
    print(f"Output: V={result.voltage}, I={result.current}, P={result.power}")

    print("\nDone.")
