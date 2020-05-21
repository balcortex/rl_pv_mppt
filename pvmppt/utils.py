import numpy as np


def clip_var(self, value, minimum=-np.inf, maximum=np.inf):
    return min(max(value, minimum), maximum)
