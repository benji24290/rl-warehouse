try:
    from envs.warehouse_env_dir.warehouse_env import WarehouseEnv
    from envs.warehouse_env_dir.env_config import EnvConfig
    from envs.warehouse_env_dir.heuristic_agent import heuristic
    from envs.warehouse_env_dir.q_learning_agent2 import run_q_learning_agent
    from env.warehouse_env_dir.sarsa_agent import run_sarsa_agent
    from env.warehouse_env_dir.policy_test_agent import run_policy_test_agent
except ModuleNotFoundError:
    from warehouse_env import WarehouseEnv
    from env_config import EnvConfig
    from heuristic_agent import heuristic
    from q_learning_agent2 import run_q_learning_agent
    from sarsa_agent import run_sarsa_agent
    from policy_test_agent import run_policy_test_agent
    import plotting


import matplotlib.pyplot as plt
# import seaborn as sns
import math
# import pandas as pd
import numpy as np

'''
def plot_rewards(array, label, window=20):
    time_series_df = pd.DataFrame(
        array)
    smooth_path = time_series_df.rolling(window).mean()
    path_deviation = 2 * time_series_df.rolling(window).std()
    plt.plot(smooth_path, label=label, linewidth=2)
    plt.fill_between(path_deviation.index, (smooth_path-2*path_deviation)
                     [0], (smooth_path+2*path_deviation)[0], alpha=.1)
'''


# Epsilon greedy action selection
eps_decay_factor = 0.9999  # After every episode, eps is 0.9 times the previous one
eps_min = 0.05  # 10% exploration is compulsory till the end

num_episodes = 5000
alpha = 0.6
gamma = 0.999

random_seed = 60
config = EnvConfig(seed=1234,  turns=100,
                   steps_to_request=4)


compare = False

if(compare):
    # Learn Policies
    results_sarsa = run_sarsa_agent(WarehouseEnv(
        config), num_episodes, alpha, gamma, eps_decay_factor, random_seed)

    results_q_learning = run_q_learning_agent(WarehouseEnv(
        config), num_episodes, alpha, gamma, eps_decay_factor, random_seed)

    # Test learned Policies
    results_q_learning_policy = run_policy_test_agent(env=WarehouseEnv(
        config), num_episodes=1000, Q=results_q_learning.q, random_seed=random_seed)
    results_sarsa_policy = run_policy_test_agent(env=WarehouseEnv(
        config), num_episodes=1000, Q=results_sarsa.q, random_seed=random_seed)

    rew_h_v4 = heuristic(config, count=1000, version='v4')

    plt.xlabel('Epochen')
    plt.ylabel('∅-Reward pro Step')
    plt.title('Bestellung alle 4 Steps')

    results_q_learning.plot_episode_rewards(
        label='q-learn', window=50, std=False)
    results_sarsa.plot_episode_rewards(
        label='sarsa', window=50, std=False)
    results_q_learning_policy.plot_episode_rewards(
        label='q-final-policy', window=50, std=False)
    results_sarsa_policy.plot_episode_rewards(
        label='sarsa-final-policy', window=50, std=False)
    rew_h_v4.plot_episode_rewards(label='heur-v4', window=50, std=False)

    plt.legend()
    plt.show()

    # so für alle verschiedenen parameter zum besser darstellen
    plt.xlabel('Epochen')
    plt.ylabel('∅-Reward pro Step')
    plt.title('Bestellung alle 4 Steps')

    results_q_learning.plot_episode_rewards(label='q-learn', window=50)
    results_sarsa.plot_episode_rewards(label='sarsa', window=50)
    results_q_learning_policy.plot_episode_rewards(
        label='q-final-policy', window=50)
    results_sarsa_policy.plot_episode_rewards(
        label='sarsa-final-policy', window=50)
    rew_h_v4.plot_episode_rewards(label='heur-v4', window=50)

    plt.legend()
    plt.show()

    plt.xlabel('xxxxxx')
    plt.ylabel('yyyyy')
    plt.title('Squared TD Error')
    # should not be in same graph and should be log
    results_sarsa.plot_squared_td_errors(
        label='squared tderror', std=False)
    plt.legend()
    plt.show()

    results_sarsa.plot_exploration()
