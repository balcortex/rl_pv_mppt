# pylint: disable=no-value-for-parameter

import argparse
import logging
import os
import time
from pathlib import Path

import gin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import batched_py_environment, suite_gym, tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver, random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from pvenv_disc import PVEnv


@gin.configurable
class ReinforceTrainer:
    def __init__(
        self,
        root_dir,
        train_env,
        eval_env,
        optimizer,
        fc_layer_params=(100,),
        num_eval_episodes=None,
        num_iterations=10_000,
        replay_buffer_capacity=100_000,
        sample_batch_size=64,
        n_step_update=1,
        initial_collect_steps=1_000,
        collect_steps_per_iteration=1,
        summary_interval=100,
        train_steps_per_iteration=1,
        log_interval=100,
        train_checkpoint_interval=1_000,
        policy_checkpoint_interval=1_000,
        rb_checkpoint_interval=1_000,
        eval_interval=1_000,
        lr_decay_fraction_step=0.8,
        epsilon_greedy=0.1,
    ):
        assert isinstance(
            train_env, tf_py_environment.TFPyEnvironment
        ), "Only TFPyEnvironment envs are supported"
        assert isinstance(
            eval_env, tf_py_environment.TFPyEnvironment
        ), "Only TFPyEnvironment envs are supported"

        Path(root_dir).mkdir(exist_ok=True, parents=True)

        self.train_dir = os.path.join(root_dir, "train")
        self.eval_dir = os.path.join(root_dir, "eval")
        self.policy_dir = os.path.join(self.train_dir, "policy")

        self.train_env = train_env
        self.eval_env = eval_env
        self._num_eval_episodes = num_eval_episodes
        self._num_iterations = num_iterations
        self._replay_buffer_capacity = replay_buffer_capacity
        self._sample_batch_size = sample_batch_size
        self._n_step_update = n_step_update
        self._initial_collect_steps = initial_collect_steps
        self._collect_steps_per_iteration = collect_steps_per_iteration
        self._summary_interval = summary_interval
        self._train_steps_per_iteration = train_steps_per_iteration
        self._log_interval = log_interval
        self._train_checkpoint_interval = train_checkpoint_interval
        self._policy_checkpoint_interval = policy_checkpoint_interval
        self._rb_checkpoint_interval = rb_checkpoint_interval
        self._eval_interval = eval_interval

        self.train_summary_writer = tf.compat.v2.summary.create_file_writer(
            self.train_dir
        )
        self.train_summary_writer.set_as_default()
        self.eval_summary_writer = tf.compat.v2.summary.create_file_writer(
            self.eval_dir
        )
        self.train_metrics = [
            tf_metrics.AverageReturnMetric(
                batch_size=train_env.batch_size, buffer_size=1
            ),
            tf_metrics.AverageEpisodeLengthMetric(
                batch_size=train_env.batch_size, buffer_size=1
            ),
            # tf_metrics.EnvironmentSteps(),
            # tf_metrics.NumberOfEpisodes(),
        ]
        self.eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=self._num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=self._num_eval_episodes),
            # tf_metrics.EnvironmentSteps(),
            # tf_metrics.NumberOfEpisodes(),
        ]
        self.train_step_counter = tf.Variable(
            0, dtype=tf.int64
        )  # network update counter
        self.agent = None
        self._fc_layet_params = fc_layer_params
        self._optimizer = optimizer
        self._lr_decay_fraction_step = lr_decay_fraction_step
        self._epsilon_greedy = epsilon_greedy
        self.replay_buffer = None
        self.iterator = None
        self.random_policy = None
        self.initial_collect_driver = None
        self.collect_driver = None
        self.train_checkpointer = None
        self.policy_checkpointer = None
        self.rb_checkpointer = None

        print(f'Train dir: "{self.train_dir}"')
        print(f'Eval dir:  "{self.eval_dir}"')
        print()
        print(f"Train env batch size: {train_env.batch_size}")
        print(f"Eval env batch size: {eval_env.batch_size}")

    def init_dqn_agent(self,):
        assert (
            self.train_env.action_spec().dtype.is_integer
        ), "Only discrete envs are supported"
        assert (
            self.eval_env.action_spec().dtype.is_integer
        ), "Only discrete envs are supported"

        q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=self._fc_layet_params,
        )
        epsilon_greedy_fn = (
            tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=1.0,
                decay_steps=self._lr_decay_fraction_step * self._num_iterations,
                end_learning_rate=0.01,
            )
            if self._lr_decay_fraction_step
            else lambda x: self._epsilon_greedy
        )
        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=q_net,
            optimizer=self._optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter,
            epsilon_greedy=lambda: epsilon_greedy_fn(self.train_step_counter),
        )
        self.agent.initialize()
        print("DQN Agent initialized succesfully.")

        self.init_replay_buffer()
        self.init_policies()
        self.init_drivers()
        self.init_checkpoints()
        self._load_checkpoints()

    def init_replay_buffer(self):
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self._replay_buffer_capacity,
        )
        dataset = self.replay_buffer.as_dataset(
            sample_batch_size=self._sample_batch_size,
            num_steps=self._n_step_update + 1,
            num_parallel_calls=3,
        ).prefetch(3)
        self.iterator = iter(dataset)
        print("Replay buffer initialized succesfully.")

    def init_policies(self):
        self.random_policy = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(), self.train_env.action_spec()
        )
        print("Policies initialized succesfully.")

    def init_drivers(self):
        self.initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            env=self.train_env,
            policy=self.random_policy,
            observers=[self.replay_buffer.add_batch] + self.train_metrics,
            num_steps=self._initial_collect_steps,
        )
        self.collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
            self.agent.collect_policy,
            observers=[self.replay_buffer.add_batch] + self.train_metrics,
            num_steps=self._collect_steps_per_iteration,
        )
        print("Drivers initialized succesfully.")

    def init_checkpoints(self):
        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.train_dir,
            agent=self.agent,
            global_step=self.train_step_counter,
            max_to_keep=1,  # oldest checkpoints are deleted
            metrics=metric_utils.MetricsGroup(self.train_metrics, "train_metrics"),
        )
        self.policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(self.train_dir, "policy"),
            # policy=self.agent.policy,
            policy=self.agent.policy,
            collect_policy=self.agent.collect_policy,
            max_to_keep=1,  # oldest checkpoints are deleted
            global_step=self.train_step_counter,
        )
        self.rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(self.train_dir, "replay_buffer"),
            max_to_keep=1,  # oldest checkpoints are deleted
            replay_buffer=self.replay_buffer,
        )
        self.tf_policy_saver = policy_saver.PolicySaver(self.agent.policy)

    def _load_checkpoints(self):
        self.train_checkpointer.initialize_or_restore().expect_partial()
        self.rb_checkpointer.initialize_or_restore().expect_partial()
        self.policy_checkpointer.initialize_or_restore().expect_partial()

        print("Checkpoints initialized succesfully.")

    def _init_train(self):
        self.agent.train = common.function(self.agent.train)
        self.collect_driver.run = common.function(self.collect_driver.run)

        if not self.rb_checkpointer.checkpoint_exists:
            print(
                "Initializing replay buffer by collecting experience with a random policy . . ."
            )
            self.initial_collect_driver.run()
        else:
            print("Loading Replay Buffer . . .")

        # # Evaluate the agent's policy before training.
        # metric_utils.eager_compute(
        #     metrics=self.eval_metrics,
        #     environment=self.eval_env,
        #     policy=self.agent.policy,
        #     num_episodes=self._num_eval_episodes,
        #     train_step=self.train_step_counter,
        #     summary_writer=self.eval_summary_writer,
        #     summary_prefix="Metrics",
        # )
        # metric_utils.log_metrics(self.eval_metrics)
        # print(
        #     # "Avg Return: {:,.4f}; Avg Length: {:,.4f}".format(
        #     #     eval_metrics[0].result(), eval_metrics[1].result()
        #     # )
        # )

    def train(self):
        self._init_train()

        # Initial time_step and policy_state
        time_step = None
        policy_state = self.agent.collect_policy.get_initial_state(
            self.train_env.batch_size
        )

        # Variables to measure elapsed time
        timed_at_step = self.train_step_counter.numpy()
        time_acc = 0

        # Write summary every 'summary_interval' steps
        print("Beginning training")
        with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(self.train_step_counter % self._summary_interval, 0)
        ):

            for _ in range(self._num_iterations):
                start_time = time.time()

                # Run the collect driver and save trajectories to the replay buffer
                time_step, policy_state = self.collect_driver.run(
                    time_step=time_step, policy_state=policy_state
                )
                # Train agent's network
                for _ in range(self._train_steps_per_iteration):
                    experience, _ = next(self.iterator)
                    train_loss = self.agent.train(experience)
                time_acc += time.time() - start_time

                step = self.train_step_counter.numpy()
                if step % self._log_interval == 0:
                    eps = self.agent._epsilon_greedy()
                    # eps = self.agent._epsilon_greedy
                    steps_per_sec = (step - timed_at_step) / time_acc
                    print(
                        "Step: {:,d}; Loss: {:,.4f}; Eps: {:.4f}; Steps/sec: {:.1f}".format(
                            step, train_loss.loss, eps, steps_per_sec
                        )
                    )
                    tf.compat.v2.summary.scalar(
                        name="global_steps_per_sec", data=steps_per_sec, step=step
                    )
                    tf.compat.v2.summary.scalar(name="epsilon", data=eps, step=step)
                    timed_at_step = step
                    time_acc = 0

                for train_metric in self.train_metrics:
                    train_metric.tf_summaries(
                        train_step=step, step_metrics=self.train_metrics
                    )

                if step % self._train_checkpoint_interval == 0:
                    self.train_checkpointer.save(global_step=step)

                if step % self._policy_checkpoint_interval == 0:
                    self.policy_checkpointer.save(global_step=step)
                    self.tf_policy_saver.save(self.policy_dir)

                if step % self._rb_checkpoint_interval == 0:
                    self.rb_checkpointer.save(global_step=step)

                if step % self._eval_interval == 0:
                    metric_utils.eager_compute(
                        metrics=self.eval_metrics,
                        environment=self.eval_env,
                        policy=self.agent.policy,
                        num_episodes=self._num_eval_episodes,
                        train_step=self.train_step_counter,
                        summary_writer=self.eval_summary_writer,
                        summary_prefix="Metrics",
                    )
                    metric_utils.log_metrics(self.eval_metrics)
                    # print(
                    #     # "Avg Return: {:,.4f}; Avg Length: {:,.4f}".format(
                    #     #     eval_metrics[0].result(), eval_metrics[1].result()
                    # )


gin.external_configurable(tf.compat.v1.train.AdamOptimizer)


@gin.configurable
def run_rl(
    root_dir,
    pv_array_model,
    pv_model_path,
    pv_weather_db_path,
    max_episode_steps_train=None,
    max_episode_steps_eval=None,
    num_discrete_actions=3,
    voltage_delta=0.1,
    num_batched_train_envs=1,
    v0=25,
):
    if isinstance(pv_model_path, (list, tuple)):
        pv_model_path = os.path.join(*pv_model_path)
    if isinstance(pv_weather_db_path, (list, tuple)):
        pv_weather_db_path = os.path.join(*pv_weather_db_path)

    tf_train_env = tf_py_environment.TFPyEnvironment(
        batched_py_environment.BatchedPyEnvironment(
            [
                PVEnv(
                    pv_array_model,
                    pv_model_path,
                    pv_weather_db_path,
                    max_episode_steps=max_episode_steps_train,
                    num_discrete_actions=num_discrete_actions,
                    delta_v=voltage_delta,
                    v0=v0,
                )
                for _ in range(num_batched_train_envs)
            ]
        )
    )
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

    rtrainer = ReinforceTrainer(root_dir, tf_train_env, tf_eval_env)
    rtrainer.init_dqn_agent()
    rtrainer.train()


def parse_args():
    parser = argparse.ArgumentParser(
        description="DQN Agent to solve the MPP problem for a discrete environment"
    )
    parser.add_argument(
        "rootdir",
        type=str,
        help="root dir for retrieve parameters, store checkpoints and training data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    args = parse_args()

    for dir_ in args.rootdir.split(","):

        gin.parse_config_file(os.path.join(dir_, "config_run.gin"))

        with tf.device("/cpu:0"):
            run_rl(root_dir=dir_)

    print("\n\n D O N E")
