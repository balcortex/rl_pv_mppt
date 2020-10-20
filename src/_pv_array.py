import os
from collections import namedtuple
from functools import lru_cache, partial
from typing import Dict, List, NamedTuple

import matlab.engine
from scipy.optimize import minimize

from src.matlab_api import set_parameters
from src.logger import logger

PVSimResult = namedtuple("PVSimResult", ["power", "voltage", "current"])

PV_ARRAY_DEFAULT_PARAMETERS = {
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


class PVArray:
    def __init__(
        self, model_params: Dict = PV_ARRAY_DEFAULT_PARAMETERS, float_precision: int = 3
    ) -> None:
        """PV Array Model, interface between MATLAB and Python

        Params:
            model_params: dictionary with the array parameters
            float_precision: decimal places used by the model (for cache)
        """
        logger.info("Starting MATLAB engine . . .")
        self.model_params = model_params
        self.voc = float(model_params["Voc"])
        self.eng = matlab.engine.start_matlab()

        self._model_path = os.path.join(".", "src", "matlab_model")
        self._model_name = os.path.basename(self._model_path)
        self._float_precision = float_precision
        self._running = False

        self._init()

    def __repr__(self) -> str:
        return str(self.model_params)

    def step(
        self, load_voltage: float, irradiance: float, cell_temp: float
    ) -> PVSimResult:
        "Simulate one step on Simulink"
        return self._step_cached(
            round(load_voltage, self._float_precision),
            round(irradiance, self._float_precision),
            round(cell_temp, self._float_precision),
        )

    @lru_cache(maxsize=None)
    def _step_cached(self, v: float, g: float, t: float) -> PVSimResult:
        "Auxiliar function to cache results from the simulation (speed)"
        if not self._running:
            self._prepare_for_step()

        self._set_irradiance(g)
        self._set_cell_temp(t)
        self._set_load_voltage(v)

        set_parameters(
            self.eng, self._model_name, {"SimulationCommand": ["continue", "pause"]}
        )

        # Get results from MATLAB workspace
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
        "Simulate multiple steps for value stability"
        return self._simulate_cached(
            round(load_voltage, self._float_precision),
            round(irradiance, self._float_precision),
            round(cell_temp, self._float_precision),
        )

    @lru_cache(maxsize=None)
    def _simulate_cached(self, v: float, g: float, t: float) -> PVSimResult:
        "Auxilar function to cache the results from the simulation (speed)"
        logger.debug(f"Simulate called with args: V={v}, G={g}, T={t}")

        if self._running:
            self._prepare_for_sim()

        self._set_irradiance(g)
        self._set_cell_temp(t)
        self._set_load_voltage(v)

        set_parameters(self.eng, self._model_name, {"SimulationCommand": "start"})

        # self.eng.eval("pause(0.1)", nargout=0)

        power = self.eng.eval("P(end);", nargout=1)
        voltage = self.eng.eval("V(end);", nargout=1)
        current = self.eng.eval("I(end);", nargout=1)

        logger.debug(
            f"Simulation completed, results:  P={power:.3f}, V={voltage:.3f}, I={current:.3f}"
        )
        return PVSimResult(
            round(power, self._float_precision),
            round(voltage, self._float_precision),
            round(current, self._float_precision),
        )

    def get_true_mpp(
        self, irradiance: List[float], temperature: List[float], ftol: float = 1e-06
    ) -> PVSimResult:
        """Get the real MPP for the specified inputs

        Params:
            irradiance: solar irradiance [w/m^2]
            temperature: cell temperature [celsius]
            ftol: tolerance of the solver (optimizer)
        """
        logger.info("Getting true MPP . . .")
        float_precision = self._float_precision
        self._float_precision = 8
        voltages, powers, currents = [], [], []

        for (idx), (g, t) in enumerate(zip(irradiance, temperature)):
            logger.debug(f"Calculating true MPP for index: {idx}/{len(irradiance) - 1}")

            # Call with fewer decimal places precision
            result = self._get_true_mpp_cached(
                round(g, float_precision), round(t, float_precision), ftol
            )

            voltages.append(round(result.voltage, float_precision))
            powers.append(round(result.power, float_precision))
            currents.append(round(result.current, float_precision))
            logger.debug(f"MPP = {powers[-1]}, G = {g}, T = {t}")

        self._float_precision = float_precision

        if len(powers) == 1:
            return PVSimResult(powers[0], voltages[0], currents[0])
        return PVSimResult(powers, voltages, currents)

    @lru_cache(maxsize=None)
    def _get_true_mpp_cached(self, g: float, t: float, ftol: float) -> PVSimResult:
        "Auxiliar function to cache the results from the MPP simulation"
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
        delta_v: float = 0.1,
    ) -> PVSimResult:
        """
        Perform the P&O MPPT technique

        Params:
            irradiance: solar irradiance [W/m^2]
            temperature: pv array temperature [celsius]
            v0: initial voltage of the load
            delta_v: delta voltage for incrementing/decrementing the load voltage

        """
        logger.info(f"Running P&O, step = {delta_v} volts . . .")
        voltages, currents, powers = [], [], []
        p0 = 0
        v = 0.0

        for (idx), (g, t) in enumerate(zip(irradiance, temperature)):
            logger.debug(f"Calculating P&O MPP for index: {idx}/{len(irradiance) - 1}")

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
                            v += delta_v
                        else:
                            v -= delta_v
                    else:
                        if delta_v > 0:
                            v -= delta_v
                        else:
                            v += delta_v

            voltages.append(round(v0, self._float_precision))
            currents.append(round(i, self._float_precision))
            powers.append(round(p, self._float_precision))
            logger.debug(f"MPP = {powers[-1]}, G = {g}, T = {t}")

        return PVSimResult(powers, voltages, currents)

    def _init(self) -> None:
        "Load the model and initialize it"
        self.eng.eval("beep off", nargout=0)
        self.eng.eval('model = "{}";'.format(self._model_path), nargout=0)
        self.eng.eval("load_system(model)", nargout=0)

        set_parameters(self.eng, [self._model_name, "PV Array"], self.model_params)
        logger.info("Model loaded succesfully.")

    def _prepare_for_step(self) -> None:
        "Prepare the model for taking steps"
        set_parameters(
            self.eng, self._model_name, {"SimulationCommand": ["start", "pause"]}
        )
        set_parameters(self.eng, self._model_name, {"StopTime": "inf"})
        self._running = True

    def _prepare_for_sim(self) -> None:
        "Prepare the model for a complete simulation"
        set_parameters(self.eng, self._model_name, {"SimulationCommand": "stop"})
        set_parameters(self.eng, self._model_name, {"StopTime": "1e-3"})
        self._running = False

    def _quit_engine(self) -> None:
        "Quit the MATLAB engine"
        logger.info("Quitting engine . . .")
        self.eng.quit()

    def _get_negative_output_power(
        self, load_voltage: List[float], irradiance: float, cell_temp: float
    ) -> float:
        """
        Auxiliar function for finding the real MPP.
        Because the optimization process is a minimization, the output must be negative.
        """
        sim_result = self.simulate(load_voltage[0], irradiance, cell_temp)
        return -sim_result.power

    def _set_load_voltage(self, voltage: float) -> None:
        "Auxiliar function for setting the load voltage source on the Simulink model"
        set_parameters(
            self.eng,
            [self._model_name, "Variable DC Source", "Load Voltage"],
            {"Value": str(voltage)},
        )

    def _set_irradiance(self, irradiance: float) -> None:
        "Auxiliar function for setting the irradiance on the Simulink model"
        set_parameters(
            self.eng, [self._model_name, "Irradiance"], {"Value": str(irradiance)}
        )

    def _set_cell_temp(self, cell_temp: float) -> None:
        "Auxiliar function for setting the cell temperature on the Simulink model"
        set_parameters(
            self.eng, [self._model_name, "Cell Temperature"], {"Value": str(cell_temp)}
        )


if __name__ == "__main__":

    pv_array = PVArray(float_precision=8)

    result = pv_array.simulate(28.459, 1000, 25)
    print("Simulation")
    print(f"Inputs: V={28.459}, G={1000}, T={25}")
    print(f"Output: V={result.voltage}, I={result.current}, P={result.power}")

    result = pv_array.get_true_mpp([1000], [25])
    print("MPP Simulation")
    print(f"Inputs: G={1000}, T={25}")
    print(f"Output: V={result.voltage}, I={result.current}, P={result.power}")

    print("\nDone.")
