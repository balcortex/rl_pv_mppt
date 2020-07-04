import os
import time
import logging
from typing import Optional, Sequence

# import gin
# from six.moves import range
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

from pvmppt.data_manager import parse_depfie_csv
from pvmppt.pv_array import PVArray
from pvmppt.pvenv import PVEnvDiscFullV0

tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name="Adam",
)


def train_eval(
    root_dir: str,
    num_iterations: int,
    # Params for QNetwork
    fc_layer_params: Sequence[int] = (20,),
    # Params for collect
    initial_collect_steps: int = 1000,
    collect_steps_per_iteration: int = 1,
    epsilon_greedy: float = 0.1,
    replay_buffer_capacity: int = 10_000,
    # Params for target update
    target_update_tau: float = 0.05,
    target_update_period: int = 5,
    # Params for train
    train_steps_per_iteration: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    discount: float = 0.99,
    reward_scale_factor: float = 1.0,
    gradient_clipping: bool = False,
    use_tf_functions: bool = True,
    # Params for eval
    num_eval_episodes: int = 1,
    eval_interval: int = 1000,
    # Params for checkpoints
    train_checkpoint_interval: int = 10_000,
    policy_checkpoint_interval: int = 5000,
    rb_checkpoint_interval: int = 20_000,
    # Params for summaries and loggging
    log_interval: int = 1000,
    summary_interval: int = 1000,
    debug_summaries: bool = False,
    summarize_grads_and_vars: bool = False,
    eval_metrics_callback=None,
):
    """A simple train and eval for DQN."""
    # path
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, "train")
    eval_dir = os.path.join(root_dir, "eval")

    # summaries
    train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir)
    train_summary_writer.set_as_default()
    eval_summary_writer = tf.compat.v2.summary.create_file_writer(eval_dir)

    # metrics
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
    ]
    train_metrics = [
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    # pvarray
    pvarray = PVArray(
        {
            "Npar": "1",
            "Nser": "1",
            "Ncell": "54",
            "Voc": "32.9",
            "Isc": "8.21",
            "Vm": "26.3",
            "Im": "7.61",
            "beta_Voc_pc": "-0.1230",
            "alpha_Isc_pc": "0.0032",
            "BAL": "on",
            "Tc": "1e-6",
        },
        float_precision=3,
    )
    weather_df = parse_depfie_csv(os.path.join("data", "toy_weather.csv"))

    global_step = tf.compat.v1.train.get_or_create_global_step()

    # environments
    tf_env = tf_py_environment.TFPyEnvironment(
        PVEnvDiscFullV0(
            pvarray=pvarray,
            weather_df=weather_df,
            v_delta=0.1,
            discount=discount,
            # v0=25,
        )
    )
    eval_tf_env = tf_env

    # network
    q_net = q_network.QNetwork(
        input_tensor_spec=tf_env.observation_spec(),
        action_spec=tf_env.action_spec(),
        fc_layer_params=fc_layer_params,
    )

    # agent
    tf_agent = dqn_agent.DqnAgent(
        time_step_spec=tf_env.time_step_spec(),
        action_spec=tf_env.action_spec(),
        q_network=q_net,
        epsilon_greedy=epsilon_greedy,
        n_step_update=1,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=discount,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step,
    )
    tf_agent.initialize()

    # policies
    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec()
    )

    # replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity,
    )

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2,
    ).prefetch(3)
    iterator = iter(dataset)

    # drivers
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=collect_steps_per_iteration,
    )

    # checkpoints
    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, "train_metrics"),
    )
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, "policy"),
        policy=eval_policy,
        global_step=global_step,
    )
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, "replay_buffer"),
        max_to_keep=1,
        replay_buffer=replay_buffer,
    )
    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()
    policy_checkpointer.initialize_or_restore()

    def train_step():
        experience, _ = next(iterator)
        return tf_agent.train(experience)

    # To speed up collect use common.function
    if use_tf_functions:
        collect_driver.run = common.function(collect_driver.run)
        tf_agent.train = common.function(tf_agent.train)
        train_step = common.function(train_step)

    # Collect initial replay data
    logging.info(
        f"Initializing replay buffer by collecting experience for {initial_collect_steps} steps with a random policy."
    )
    dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=initial_collect_steps,
    ).run()

    # Evaluate policy before training
    results = metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix="Metrics",
    )
    if eval_metrics_callback is not None:
        eval_metrics_callback(results, global_step.numpy())
    metric_utils.log_metrics(eval_metrics)

    # train
    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)
    timed_at_step = global_step.numpy()
    time_acc = 0
    for _ in range(num_iterations):
        start_time = time.time()
        time_step, policy_state = collect_driver.run(
            time_step=time_step, policy_state=policy_state,
        )
        for _ in range(train_steps_per_iteration):
            train_loss = train_step()
        time_acc += time.time() - start_time

        if global_step.numpy() % log_interval == 0:
            logging.info(f"step = {global_step.numpy()}, loss = {train_loss.loss}")
            steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
            logging.info(f"{steps_per_sec:.3f} steps/sec")
            tf.compat.v2.summary.scalar(
                name="global_steps_per_sec", data=steps_per_sec, step=global_step
            )
            timed_at_step = global_step.numpy()
            time_acc = 0

        if global_step.numpy() % summary_interval == 0:
            for train_metric in train_metrics:
                train_metric.tf_summaries(
                    train_step=global_step,
                    # step_metrics=train_metrics[:2]
                )

        if global_step.numpy() % train_checkpoint_interval == 0:
            train_checkpointer.save(global_step=global_step.numpy())

        if global_step.numpy() % policy_checkpoint_interval == 0:
            policy_checkpointer.save(global_step=global_step.numpy())

        if global_step.numpy() % rb_checkpoint_interval == 0:
            rb_checkpointer.save(global_step=global_step.numpy())

        if global_step.numpy() % eval_interval == 0:
            results = metric_utils.eager_compute(
                eval_metrics,
                eval_tf_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix="Metrics",
            )
            if eval_metrics_callback is not None:
                eval_metrics_callback(results, global_step.numpy())
            metric_utils.log_metrics(eval_metrics)

    return train_loss


def main():
    logging.basicConfig(level=logging.INFO)
    # tf.compat.v1.enable_v2_behavior()
    # gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    # root_dir = os.path.join("trained_agents", "dqn", "001")
    train_eval(root_dir="run_test", num_iterations=200_000)


if __name__ == "__main__":
    main()
