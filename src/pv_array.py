import os
from collections import namedtuple
from functools import partial
from typing import Dict, List, Union

import matlab.engine
from scipy.optimize import minimize
from tqdm import tqdm

from src import utils
from src.logger import logger
from src.matlab_api import set_parameters

PVSimResult = namedtuple("PVSimResult", ["power", "voltage", "current"])


class PVArray:
    def __init__(self, params: Dict):
        """PV Array Model, interface between MATLAB and Python

        Params:
            model_params: dictionary with the parameters
        """
        logger.info("Starting MATLAB engine . . .")
        self._params = params
        self._eng = matlab.engine.start_matlab()
        self._model_path = os.path.join("src", "matlab_model")

        self._init()

    def __repr__(self) -> str:
        return (
            f"PVArray {float(self.params['Im']) * float(self.params['Vm']):.0f} Watts"
        )

    def simulate(
        self, voltage: float, irradiance: float, cell_temp: float
    ) -> PVSimResult:
        """
        Simulate the simulink model

        Params:
            votlage: load voltage [V]
            irradiance: solar irradiance [W/m^2]
            temperature: cell temperature [celsius]
        """
        self._set_voltage(voltage)
        self._set_irradiance(irradiance)
        self._set_cell_temp(cell_temp)
        self._start_simulation()

        pv_power = self._eng.eval("P(end);", nargout=1)
        pv_voltage = self._eng.eval("V(end);", nargout=1)
        pv_current = self._eng.eval("I(end);", nargout=1)

        return PVSimResult(pv_power, pv_voltage, pv_current)

    def get_true_mpp(
        self,
        irradiance: Union[float, List[float]],
        cell_temp: Union[float, List[float]],
        ftol: float = 1e-06,
    ) -> PVSimResult:
        """Get the real MPP for the specified inputs

        Params:
            irradiance: solar irradiance [w/m^2]
            temperature: cell temperature [celsius]
            ftol: tolerance of the solver (optimizer)
        """
        if isinstance(irradiance, (int, float)):
            irradiance = [irradiance]
            cell_temp = [cell_temp]
        assert isinstance(irradiance[0], (int, float))
        assert isinstance(cell_temp[0], (int, float))
        assert len(cell_temp) == len(
            irradiance
        ), "irradiance and cell_temp lists must be the same length"

        logger.info("Calculating true MPP . . .")
        pv_voltages, pv_powers, pv_currents = [], [], []

        # Auxiliar function to maximize the power (scipy performs minimization)
        neg_power_fn = lambda v, g, t: self.simulate(v[0], g, t)[0] * -1
        for g, t in tqdm(
            list(zip(irradiance, cell_temp)),
            desc="Calculating true MPP",
            ascii=True,
        ):
            min_fn = partial(neg_power_fn, g=g, t=t)
            optim_result = minimize(
                min_fn, 0.8 * self.voc, method="SLSQP", options={"ftol": ftol}
            )
            assert optim_result.success == True
            pv_voltages.append(optim_result.x[0])
            pv_powers.append(optim_result.fun * -1)
            pv_currents.append(pv_powers[-1] / pv_voltages[-1])

        if len(pv_powers) == 1:
            return PVSimResult(pv_powers[0], pv_voltages[0], pv_currents[0])
        return PVSimResult(pv_powers, pv_voltages, pv_currents)

    def get_po_mpp(
        self,
        irradiance: List[float],
        cell_temp: List[float],
        v0: float = 0.0,
        v_step: float = 0.1,
    ) -> PVSimResult:
        """
        Perform the P&O MPPT technique

        Params:
            irradiance: solar irradiance [W/m^2]
            temperature: pv array temperature [celsius]
            v0: initial voltage of the load
            v_step: delta voltage for incrementing/decrementing the load voltage

        """
        assert isinstance(irradiance[0], (int, float))
        assert isinstance(cell_temp[0], (int, float))
        assert len(cell_temp) == len(
            irradiance
        ), "irradiance and cell_temp lists must be the same length"

        logger.info(f"Running P&O, step={v_step} volts . . .")
        pv_voltages, pv_powers, pv_currents = [v0, v0], [0], []

        for g, t in tqdm(
            list(zip(irradiance, cell_temp)),
            desc="Calculating PO",
            ascii=True,
        ):
            sim_result = self.simulate(pv_voltages[-1], g, t)
            delta_v = pv_voltages[-1] - pv_voltages[-2]
            delta_p = sim_result.power - pv_powers[-1]
            pv_powers.append(sim_result.power)
            pv_currents.append(sim_result.current)

            if delta_p == 0:
                pv_voltages.append(pv_voltages[-1])
            else:
                if delta_p > 0:
                    if delta_v >= 0:
                        pv_voltages.append(pv_voltages[-1] + v_step)
                    else:
                        pv_voltages.append(pv_voltages[-1] - v_step)
                else:
                    if delta_v >= 0:
                        pv_voltages.append(pv_voltages[-1] - v_step)
                    else:
                        pv_voltages.append(pv_voltages[-1] + v_step)

        return PVSimResult(pv_powers[1:], pv_voltages[1:-1], pv_currents)

    def _init(self) -> None:
        "Load the model and initialize it"
        self._eng.eval("beep off", nargout=0)
        self._eng.eval('model = "{}";'.format(self._model_path), nargout=0)
        self._eng.eval("load_system(model)", nargout=0)
        set_parameters(self._eng, self.model_name, {"StopTime": "1e-3"})
        set_parameters(self._eng, [self.model_name, "PV Array"], self.params)
        logger.info("Model loaded succesfully.")

    def _set_cell_temp(self, cell_temp: float) -> None:
        "Auxiliar function for setting the cell temperature on the Simulink model"
        set_parameters(
            self._eng, [self.model_name, "Cell Temperature"], {"Value": str(cell_temp)}
        )

    def _set_irradiance(self, irradiance: float) -> None:
        "Auxiliar function for setting the irradiance on the Simulink model"
        set_parameters(
            self._eng, [self.model_name, "Irradiance"], {"Value": str(irradiance)}
        )

    def _set_voltage(self, voltage: float) -> None:
        "Auxiliar function for setting the load voltage source on the Simulink model"
        set_parameters(
            self._eng,
            [self.model_name, "Variable DC Source", "Load Voltage"],
            {"Value": str(voltage)},
        )

    def _start_simulation(self) -> None:
        "Start the simulation command"
        set_parameters(self._eng, self.model_name, {"SimulationCommand": "start"})

    @property
    def voc(self) -> float:
        "Open-circuit voltage of the pv array"
        return float(self.params["Voc"])

    @property
    def params(self) -> Dict:
        "Dictionary containing the parameters of the pv array"
        return self._params

    @property
    def model_name(self) -> str:
        "String containing the name of the model (for running in MATLAB)"
        return os.path.basename(self._model_path)

    @classmethod
    def from_json(cls, path: str):
        "Create a PV Array from a json file containing a string with the parameters"
        return cls(params=utils.load_dict(path))
