from src.reward import DiscreteRewardDeltaPower, RewardDeltaPower
from src.common import History


def test_discrete_reward_delta_power():
    history = History()
    reward_fn = DiscreteRewardDeltaPower(2.0, 0.0, 1.0, -0.5, 0.5)
    history.dp.append(-1)
    assert reward_fn(history) == -2.0
    history.dp.append(-0.4)
    assert reward_fn(history) == 0.0
    history.dp.append(0.4)
    assert reward_fn(history) == 0.0
    history.dp.append(0.6)
    assert reward_fn(history) == 1.0


def test_reward_delta_power():
    history = History()
    reward_fn = RewardDeltaPower(2.1, 1.0)
    history.dp.append(-1.1)
    assert reward_fn(history) == -2.1 * 1.1
    history.dp.append(-0.4)
    assert reward_fn(history) == -0.4 * 2.1
    history.dp.append(0.4)
    assert reward_fn(history) == 0.4 * 1.0
    history.dp.append(0.6)
    assert reward_fn(history) == 1.0 * 0.6
