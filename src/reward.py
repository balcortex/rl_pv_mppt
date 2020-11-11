from src.common import History


class Reward:
    def __call__(self, history: History) -> float:
        raise NotImplementedError


class DiscreteRewardDeltaPower(Reward):
    """
    Returns a constant reward based on the value of the change of power

    if dp < neg_treshold:
        return a
    elif neg_treshold <= dp < pos_treshold:
        return b
    else:
        return c
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 1.0,
        neg_treshold: float = -0.1,
        pos_treshold: float = 0.1,
    ):
        self.a = abs(a)
        self.b = b
        self.c = c
        self.neg_treshold = neg_treshold
        self.pos_treshold = pos_treshold

    def __call__(self, history: History) -> float:
        dp = history.dp[-1]

        if dp < self.neg_treshold:
            return -self.a
        elif self.neg_treshold <= dp < self.pos_treshold:
            return self.b
        else:
            return self.c


class RewardDeltaPower:
    """
    Returns a reward based on the value of the change of power

    if dp < 0:
        return a * dp
    else:
        return b * dp
    """

    def __init__(self, a: float, b: float):
        self.a = abs(a)
        self.b = b

    def __call__(self, history: History) -> float:
        dp = history.dp[-1]
        if dp < 0:
            return self.a * dp
        else:
            return self.b * dp


if __name__ == "__main__":
    history = History()
    reward_fn = DiscreteRewardDeltaPower(2, 0, 1)
    history.dp.append(-1.2)
    reward_fn(history)
    history.dp.append(0)
    reward_fn(history)
    history.dp.append(0.2)
    reward_fn(history)
