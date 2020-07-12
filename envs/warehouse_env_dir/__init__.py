try:
    from envs.warehouse_env_dir.warehouse_env import WarehouseEnv
    from envs.warehouse_env_dir.env_config import EnvConfig
    from envs.warehouse_env_dir.heuristic_agent import heuristic
    from envs.warehouse_env_dir.q_learning_agent import run_q_learning_agent
    from env.warehouse_env_dir.sarsa_agent import run_sarsa_agent
    from env.warehouse_env_dir.policy_test_agent import run_policy_test_agent
    from env.warehouse_env_dir.compare_parameters import compare_parameters_q_learning, compare_parameters_sarsa
except ModuleNotFoundError:
    from warehouse_env import WarehouseEnv
    from env_config import EnvConfig
    from heuristic_agent import heuristic
    from q_learning_agent import run_q_learning_agent
    from sarsa_agent import run_sarsa_agent
    from policy_test_agent import run_policy_test_agent
    from compare_parameters import compare_parameters_q_learning, compare_parameters_sarsa


import matplotlib.pyplot as plt
import math
import numpy as np


# Epsilon greedy action selection
eps_decay_factor = 0.9999  # After every episode, eps is 0.9 times the previous one
eps_min = 0.05  # 10% exploration is compulsory till the end

num_episodes = 500
alpha = 0.6
gamma = 0.999

random_seed = 60
config = EnvConfig(seed=1234,  turns=100,
                   steps_to_request=4)


if __name__ == '__main__':
    compare_parameters_q_learning(num_episodes=num_episodes, random_seed=random_seed,
                                  config=config, eps_decay_factor=eps_decay_factor)
    compare_parameters_sarsa(num_episodes=num_episodes, random_seed=random_seed,
                             config=config, eps_decay_factor=eps_decay_factor)
