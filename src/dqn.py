import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
import os
import logging
import json

from src.pv_array import PVArray
from src.utils import read_weather_csv, save_dict, load_dict
from src.pvenv import PVEnvDiscV0, PVEnv


CONFIG_DICT = {
    # "pvarray": PVArray(),
    "root_dir": os.path.join(".", "path_test"),
    "weather_path": os.path.join("data", "toy_weather.csv"),
    # "num_iterations": 10_000,
    # # Params for QNetwork
    # "fc_layer_params": (20,),
    # # Params for collect
    # "initial_collect_steps": 1000,
    # "collect_steps_per_iteration": 1,
    # "epsilon_greedy": 0.1,
    # "replay_buffer_capacity": 10_000,
    # # Params for target update
    # "target_update_tau": 0.05,
    # "target_update_period": 5,
    # # Params for train
    # "train_steps_per_iteration": 1,
    # "batch_size": 64,
    # "learning_rate": 1e-3,
    # "discount": 0.99,
    # "reward_scale_factor": 1.0,
    # "gradient_clipping": False,
    # "use_tf_functions": True,
    # # Params for eval
    # "num_eval_episodes": 1,
    # "eval_interval": 1000,
    # # Params for checkpoints
    # "train_checkpoint_interval": 10_000,
    # "policy_checkpoint_interval": 5000,
    # "rb_checkpoint_interval": 20_000,
    # # Params for summaries and loggging
    # "log_interval": 1000,
    # "summary_interval": 1000,
    # "debug_summaries": False,
    # "summarize_grads_and_vars": False,
    # "eval_metrics_callback": None,
}


class DQN:
    """
    Deep Q-Network reinforcement learning to solve the MPPT problem for photovoltaic systems

    Arguments:
        - pvarray: photovoltaic array object
        - root_dir: path to store training data

    """

    def __init__(self, pvarray: PVArray, root_dir: str, weather_path: str) -> None:
        self._locals = locals()
        self._locals.pop("self")
        self._locals["pvarray"] = str(self._locals["pvarray"])
        for key, val in self._locals.items():
            print(key, val)

        self._prepare_dirs(root_dir)

        # self.pvenv = pvenv
        # env = PVEnvDiscV0(pvarray, weather_df, v_delta=0.1, discount=0.99)

    def _prepare_dirs(self, root_dir: str) -> None:
        self._paths = {
            "root": root_dir,
            "train": os.path.join(root_dir, "train"),
            "eval": os.path.join(root_dir, "eval"),
            "fig": os.path.join(root_dir, "fig"),
            "data": os.path.join(root_dir, "data"),
        }
        for path in self._paths.values():
            logging.info(f"Folder {path} created")
            os.makedirs(path, exist_ok=True)
        save_dict(self._locals, os.path.join(root_dir, "locals.txt"))
        loaded = load_dict(os.path.join(root_dir, "locals.txt"))
        for key, val in loaded.items():
            print(key, val)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dqn = DQN(pvarray=PVArray(), **CONFIG_DICT)
