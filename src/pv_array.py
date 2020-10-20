from typing import Dict
from src import utils
from src.logger import logger
import matlab.engine
import os
from collections import namedtuple
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

    def _init(self) -> None:
        "Load the model and initialize it"
        self._running = False
        self._eng.eval("beep off", nargout=0)
        self._eng.eval('model = "{}";'.format(self._model_path), nargout=0)
        self._eng.eval("load_system(model)", nargout=0)

        set_parameters(self._eng, [self.model_name, "PV Array"], self.params)
        logger.info("Model loaded succesfully.")

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