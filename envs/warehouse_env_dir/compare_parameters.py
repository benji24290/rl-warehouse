try:
    from envs.warehouse_env_dir.warehouse_env import WarehouseEnv
    from envs.warehouse_env_dir.q_learning_agent import run_q_learning_agent
    from env.warehouse_env_dir.sarsa_agent import run_sarsa_agent
except ModuleNotFoundError:
    from warehouse_env import WarehouseEnv
    from q_learning_agent import run_q_learning_agent
    from sarsa_agent import run_sarsa_agent


import matplotlib.pyplot as plt
import math
import numpy as np


def compare_parameters_q_learning(num_episodes, random_seed, config, eps_decay_factor, eps_min):
    results_1_1 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.9, 0.9, random_seed)
    results_1_2 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.99, 0.9, random_seed)
    results_1_3 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.8, 0.9, random_seed)
    results_1_4 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.7, 0.9, random_seed)
    results_1_5 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes, 0.3, 0.9, 0.9, random_seed)
    results_1_6 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes, 0.4, 0.9, 0.9, random_seed)
    results_1_7 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes, 0.6, 0.9, 0.9, random_seed)
    results_1_8 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes, 0.7, 0.9, 0.9, random_seed)
    results_1_9 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.9, 0.95, random_seed)
    results_1_10 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.9, 0.85, random_seed)
    results_1_11 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.9, 0.99, random_seed)
    results_1_12 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.9, 0.8, random_seed)

    plt.xlabel('Epochen')
    plt.ylabel('∅-Reward pro Step')
    plt.title('Q-Learning Rewards')
    results_1_1.plot_episode_rewards(label='1.1', window=50)
    results_1_2.plot_episode_rewards(label='1.2', window=50)
    results_1_3.plot_episode_rewards(label='1.3', window=50)
    results_1_4.plot_episode_rewards(label='1.4', window=50)
    results_1_5.plot_episode_rewards(label='1.5', window=50)
    results_1_6.plot_episode_rewards(label='1.6', window=50)
    results_1_7.plot_episode_rewards(label='1.7', window=50)
    results_1_8.plot_episode_rewards(label='1.8', window=50)
    results_1_9.plot_episode_rewards(label='1.9', window=50)
    results_1_10.plot_episode_rewards(label='1.10', window=50)
    results_1_11.plot_episode_rewards(label='1.11', window=50)
    results_1_12.plot_episode_rewards(label='1.12', window=50)
    plt.legend()
    plt.show()
    plt.xlabel('Steps')
    plt.ylabel('TD-Error')
    plt.title('Q-Learning TD-Error Window ='+str(num_episodes*2))
    results_1_1.plot_squared_td_errors(
        label='1.1', window=num_episodes*2, std=False)
    results_1_2.plot_squared_td_errors(
        label='1.2', window=num_episodes*2, std=False)
    results_1_3.plot_squared_td_errors(
        label='1.3', window=num_episodes*2, std=False)
    results_1_4.plot_squared_td_errors(
        label='1.4', window=num_episodes*2, std=False)
    results_1_5.plot_squared_td_errors(
        label='1.5', window=num_episodes*2, std=False)
    results_1_6.plot_squared_td_errors(
        label='1.6', window=num_episodes*2, std=False)
    results_1_7.plot_squared_td_errors(
        label='1.7', window=num_episodes*2, std=False)
    results_1_8.plot_squared_td_errors(
        label='1.8', window=num_episodes*2, std=False)
    results_1_9.plot_squared_td_errors(
        label='1.9', window=num_episodes*2, std=False)
    results_1_10.plot_squared_td_errors(
        label='1.10', window=num_episodes*2, std=False)
    results_1_11.plot_squared_td_errors(
        label='1.11', window=num_episodes*2, std=False)
    results_1_12.plot_squared_td_errors(
        label='1.12', window=num_episodes*2, std=False)
    plt.legend()
    plt.show()


def compare_parameters_sarsa(num_episodes, random_seed, config, eps_decay_factor, eps_min):
    results_1_1 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.9, 0.9, random_seed)
    results_1_2 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.99, 0.9, random_seed)
    results_1_3 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.8, 0.9, random_seed)
    results_1_4 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.7, 0.9, random_seed)
    results_1_5 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes, 0.3, 0.9, 0.9, random_seed)
    results_1_6 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes, 0.4, 0.9, 0.9, random_seed)
    results_1_7 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes, 0.6, 0.9, 0.9, random_seed)
    results_1_8 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes, 0.7, 0.9, 0.9, random_seed)
    results_1_9 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.9, 0.95, random_seed)
    results_1_10 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.9, 0.85, random_seed)
    results_1_11 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.9, 0.99, random_seed)
    results_1_12 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes, 0.5, 0.9, 0.8, random_seed)

    plt.xlabel('Epochen')
    plt.ylabel('∅-Reward pro Step')
    plt.title('Sarsa Rewards')
    results_1_1.plot_episode_rewards(label='1.1', window=50)
    results_1_2.plot_episode_rewards(label='1.2', window=50)
    results_1_3.plot_episode_rewards(label='1.3', window=50)
    results_1_4.plot_episode_rewards(label='1.4', window=50)
    results_1_5.plot_episode_rewards(label='1.5', window=50)
    results_1_6.plot_episode_rewards(label='1.6', window=50)
    results_1_7.plot_episode_rewards(label='1.7', window=50)
    results_1_8.plot_episode_rewards(label='1.8', window=50)
    results_1_9.plot_episode_rewards(label='1.9', window=50)
    results_1_10.plot_episode_rewards(label='1.10', window=50)
    results_1_11.plot_episode_rewards(label='1.11', window=50)
    results_1_12.plot_episode_rewards(label='1.12', window=50)
    plt.legend()
    plt.show()
    plt.xlabel('Steps')
    plt.ylabel('TD-Error')
    plt.title('Sarsa TD-Error Window ='+str(num_episodes*2))
    results_1_1.plot_squared_td_errors(
        label='1.1', window=num_episodes*2, std=False)
    results_1_2.plot_squared_td_errors(
        label='1.2', window=num_episodes*2, std=False)
    results_1_3.plot_squared_td_errors(
        label='1.3', window=num_episodes*2, std=False)
    results_1_4.plot_squared_td_errors(
        label='1.4', window=num_episodes*2, std=False)
    results_1_5.plot_squared_td_errors(
        label='1.5', window=num_episodes*2, std=False)
    results_1_6.plot_squared_td_errors(
        label='1.6', window=num_episodes*2, std=False)
    results_1_7.plot_squared_td_errors(
        label='1.7', window=num_episodes*2, std=False)
    results_1_8.plot_squared_td_errors(
        label='1.8', window=num_episodes*2, std=False)
    results_1_9.plot_squared_td_errors(
        label='1.9', window=num_episodes*2, std=False)
    results_1_10.plot_squared_td_errors(
        label='1.10', window=num_episodes*2, std=False)
    results_1_11.plot_squared_td_errors(
        label='1.11', window=num_episodes*2, std=False)
    results_1_12.plot_squared_td_errors(
        label='1.12', window=num_episodes*2, std=False)
    plt.legend()
    plt.show()
