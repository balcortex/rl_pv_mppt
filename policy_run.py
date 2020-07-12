# pylint: disable=no-value-for-parameter

import logging
import os
from collections import namedtuple
import argparse
import gin
import tensorflow as tf
from tf_agents.environments import batched_py_environment, suite_gym, tf_py_environment
from tf_agents.utils import common
import matplotlib.pyplot as plt
import pandas as pd

from pvenv_disc import PVEnv

PVSimResult = namedtuple("PVSimResult", ["power", "voltage", "current"])


def convert_to_path(list_path):
    if isinstance(list_path, (list, tuple)):
        return os.path.join(*list_path)
    return list_path


@gin.configurable
def run_test(
    pv_array_model,
    pv_model_path,
    pv_weather_db_path,
    policy_dir,
    max_episode_steps_eval,
    num_discrete_actions,
    voltage_delta,
    v0,
):
    pv_model_path = convert_to_path(pv_model_path)
    pv_weather_db_path = convert_to_path(pv_weather_db_path)
    policy_dir = convert_to_path(policy_dir)

    tf_eval_env = tf_py_environment.TFPyEnvironment(
        PVEnv(
            pv_array_model,
            pv_model_path,
            pv_weather_db_path,
            max_episode_steps=max_episode_steps_eval,
            num_discrete_actions=num_discrete_actions,
            delta_v=voltage_delta,
            v0=v0,
        )
    )

    voltage = []
    power = []
    d_voltage = []
    policy = tf.compat.v2.saved_model.load(policy_dir)

    time_step = tf_eval_env.reset()
    v, p, delta_v = time_step.observation.numpy()[0]
    voltage.append(v)
    power.append(p)
    d_voltage.append(delta_v)
    logging.debug(f"V={v}, P={p}, dV={delta_v}")

    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = tf_eval_env.step(action_step.action)
        v, p, delta_v = time_step.observation.numpy()[0]
        voltage.append(v)
        power.append(p)
        d_voltage.append(delta_v)

    current = [p / v for p, v in zip(power, voltage)]

    return PVSimResult(power, voltage, current), d_voltage


@gin.configurable
def true_mpp_path(path):
    return convert_to_path(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the saved policy in the environment to see the performance of the MPP"
    )
    parser.add_argument(
        "rootdir",
        type=str,
        help="root dir for retrieve parameters, store checkpoints and training data",
    )
    parser.add_argument("-t", "--train", action="store_true", help="Perform training")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.rootdir

    gin.parse_config_file(os.path.join(root_dir, "config_test.gin"))

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if args.train:
        with tf.device("/cpu:0"):
            result, dv = run_test()

            df_rl = pd.DataFrame(
                {
                    "Power": result.power,
                    "Voltage": result.voltage,
                    "Current": result.current,
                    "Control": dv,
                }
            )
            df_rl.to_csv(os.path.join(root_dir, "rl_mpp.csv"))

    df_rl = pd.read_csv(os.path.join(root_dir, "rl_mpp.csv"))
    df_true = pd.read_csv(true_mpp_path())

    images_path = os.path.join(root_dir, "images")
    os.makedirs(images_path, exist_ok=True)
    plot_num = str((len(os.listdir(images_path)) // 3))
    plt.plot(df_rl["Power"], label="RL")
    plt.plot(df_true["Power"], label="True")
    plt.title("Power")
    plt.legend()
    plt.savefig(os.path.join(root_dir, "images", "power_comparison" + plot_num))

    plt.clf()
    plt.plot(df_rl["Voltage"], label="RL")
    plt.plot(df_true["Voltage"], label="True")
    plt.legend()
    plt.title("Voltage")
    plt.savefig(os.path.join(root_dir, "images", "voltage_comparison" + plot_num))

    plt.clf()
    plt.plot(df_rl["Control"], label="RL")
    plt.legend()
    plt.title("Control action")
    plt.savefig(os.path.join(root_dir, "images", "control" + plot_num))

    print("\n\n D O N E")
