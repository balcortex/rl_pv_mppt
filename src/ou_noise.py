import numpy as np


class OUNoise:
    """docstring for OUNoise"""

    def __init__(self, mean: float, std: float, theta: float = 0.1, dt: float = 0.01):
        self.mean = mean
        self.theta = theta
        self.std = std
        self.dt = dt
        self.reset()

    def reset(self) -> float:
        self.state = self.mean
        return self.state

    def sample(self) -> float:
        x = self.state
        dx = (
            self.theta * (self.mean - x) * self.dt
            + self.std * np.sqrt(self.dt) * np.random.randn()
        )
        self.state = x + dx
        return self.state


class GaussianNoise:
    """docstring for OUNoise"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self) -> float:
        return np.random.normal(self.mean, self.std)


if __name__ == "__main__":
    # ou = OUNoise(mean=0.0, std=0.1, theta=0.1, dt=1)
    # states = []
    # for i in range(1000000):
    #     states.append(ou.sample())
    # import matplotlib.pyplot as plt

    # plt.plot(states)
    # plt.show()

    noise = GaussianNoise(0, 0.5)
    states = []
    for i in range(800):
        states.append(noise.sample())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()