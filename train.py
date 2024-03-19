import os
import time
import gym
import numpy as np
from absl import app, flags
from utils.dataset_utils import (D4RLDataset, ReplayBuffer)
from rlagents.sac import SAC
from rlagents.cql import CQL
from rlagents.td3 import TD3
from utils.wrappers import wrap_gym
from spinupUtils.logx import EpochLogger
from spinupUtils.run_utils import setup_logger_kwargs

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "True"

FLAGS = flags.FLAGS
flags.DEFINE_string('policy', 'iql', 'policy name.')
flags.DEFINE_string('env_name', 'halfcheetah-medium-v2', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')

flags.DEFINE_integer("utd", 1, "Update to data ratio.")
flags.DEFINE_integer("num_qs", 10, "Ensemble size of REDQ.")
flags.DEFINE_float("critic_dropout_rate", 0.01, "Critic dropout rate of DroQ.")
flags.DEFINE_float('iql_expectile', 0.7, 'Expectile of IQL.')
flags.DEFINE_float("iql_temp", 3.0, "Temperature of IQL.")
flags.DEFINE_float("min_q_weight", 5.0, "Q weight of CQL.")


def evaluate(agent, env, num_episodes):
    stats = {'return': [], 'length': []}
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
        for k in stats.keys():
            stats[k].append(info['episode'][k])
    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats


def main(_):
    logger_kwargs = setup_logger_kwargs(f"{FLAGS.policy}-{FLAGS.env_name}", FLAGS.seed,
                                        data_dir=f"data/{FLAGS.policy}")
    logger = EpochLogger(**logger_kwargs)

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env)
    env.seed(FLAGS.seed)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env)
    eval_env.seed(FLAGS.seed + 42)

    if FLAGS.policy == 'sac':
        agent = SAC.create(FLAGS.seed, env.observation_space, env.action_space)
    elif FLAGS.policy == 'redq':
        agent = SAC.create(FLAGS.seed, env.observation_space, env.action_space,
                           num_qs=FLAGS.num_qs, num_min_qs=2)
    elif FLAGS.policy == 'droq':
        agent = SAC.create(FLAGS.seed, env.observation_space, env.action_space,
                           critic_dropout_rate=FLAGS.critic_dropout_rate, critic_layer_norm=True)
    elif FLAGS.policy == 'iql':
        agent = SAC.create(FLAGS.seed, env.observation_space, env.action_space,
                           log_std_min=-5, state_dependent_std=False, tanh_squash=False,
                           iql_expectile=FLAGS.iql_expectile, iql_temp=FLAGS.iql_temp)
    elif FLAGS.policy == 'cql':
        agent = CQL.create(FLAGS.seed, env.observation_space, env.action_space, min_q_weight=FLAGS.min_q_weight)
    elif FLAGS.policy in ('td3', 'td3bc'):
        agent = TD3.create(FLAGS.seed, env.observation_space, env.action_space)
    else:
        raise ValueError(f"Invalid Policy: {FLAGS.policy}!")

    if FLAGS.policy in ('sac', 'td3', 'redq', 'droq'):
        action_dim = env.action_space.shape[0]
        replay_buffer = ReplayBuffer(env.observation_space, action_dim, 3000000)
    else:
        dataset = D4RLDataset(env)

    observation, done = env.reset(), False

    for i in range(FLAGS.max_steps + 1):
        if FLAGS.policy in ('sac', 'td3', 'redq', 'droq'):
            if i < 5000:
                action = env.action_space.sample()
            else:
                action, agent = agent.sample_actions(observation, temperature=1.0)
                if FLAGS.policy == 'td3':
                    action = (action + np.random.normal(0, scale=0.1, size=action_dim)).clip(-1, 1)

            next_observation, reward, done, info = env.step(action)
            mask = 1.0 if not done or 'TimeLimit.truncated' in info else 0.0
            replay_buffer.insert(observation, action, reward, mask, float(done), next_observation)
            observation = next_observation

            if done:
                observation, done = env.reset(), False

            if i >= 5000:
                batch = replay_buffer.sample(FLAGS.batch_size, FLAGS.utd)
                if FLAGS.policy == 'td3':
                    agent, critic_info = agent.update_q(batch)
                    if i % 2:
                        agent, actor_info = agent.update_pi(batch)
                else:
                    agent, info = agent.update(batch, FLAGS.utd)

        else:
            batch = dataset.sample(FLAGS.batch_size)
            if FLAGS.policy == 'td3bc':
                agent, critic_info = agent.update_q(batch)
                if i % 2:
                    agent, actor_info = agent.update_pi_offline(batch)
            else:
                agent, info = agent.update_offline(batch)

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)
            logger.log_tabular("AverageTestEpRet", eval_stats['return'])
            logger.log_tabular("TotalEnvInteracts", i)
            logger.dump_tabular()
            print(
                f"{FLAGS.policy}-{FLAGS.env_name}-s{FLAGS.seed}-Step={i / 1000}k-Return={eval_stats['return']}")


if __name__ == '__main__':
    app.run(main)
