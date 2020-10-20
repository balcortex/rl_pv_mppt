import torch
import matlab.engine


def test_matlab_engine():
    eng = matlab.engine.start_matlab()
    assert eng.plus(2, 3) == 5

    eng.workspace["y"] = 4.0
    assert eng.eval("sqrt(y)", nargout=1) == 2.0
