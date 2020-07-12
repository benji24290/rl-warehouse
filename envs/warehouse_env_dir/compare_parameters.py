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


def compare_parameters_q_learning(num_episodes, random_seed, config, eps_decay_factor):
    results_1_1 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.9,  eps_decay_factor=0.999, random_seed=random_seed)
    results_1_2 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.99, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_3 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.8, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_4 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.7, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_5 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.3, gamma=0.9, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_6 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.4, gamma=0.9, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_7 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.6, gamma=0.9, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_8 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.7, gamma=0.9, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_9 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.9, eps_decay_factor=0.9995, random_seed=random_seed)
    results_1_10 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.9, eps_decay_factor=0.9985, random_seed=random_seed)
    results_1_11 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.9, eps_decay_factor=0.9999, random_seed=random_seed)
    results_1_12 = run_q_learning_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.9, eps_decay_factor=0.9980, random_seed=random_seed)

    plt.xlabel('Epochen')
    plt.ylabel('∅-Reward pro Step')

    window_rewards = results_1_1.plot_episode_rewards(
        label='1.1', window=int(num_episodes/100))
    results_1_2.plot_episode_rewards(label='1.2', window=int(num_episodes/100))
    results_1_3.plot_episode_rewards(label='1.3', window=int(num_episodes/100))
    results_1_4.plot_episode_rewards(label='1.4', window=int(num_episodes/100))
    results_1_5.plot_episode_rewards(label='1.5', window=int(num_episodes/100))
    results_1_6.plot_episode_rewards(label='1.6', window=int(num_episodes/100))
    results_1_7.plot_episode_rewards(label='1.7', window=int(num_episodes/100))
    results_1_8.plot_episode_rewards(label='1.8', window=int(num_episodes/100))
    results_1_9.plot_episode_rewards(label='1.9', window=int(num_episodes/100))
    results_1_10.plot_episode_rewards(
        label='1.10', window=int(num_episodes/100))
    results_1_11.plot_episode_rewards(
        label='1.11', window=int(num_episodes/100))
    results_1_12.plot_episode_rewards(
        label='1.12', window=int(num_episodes/100))
    plt.title('Q-Learning Rewards Window=' +
              str(window_rewards))
    plt.legend()
    plt.show()
    plt.xlabel('Steps')
    plt.ylabel('TD-Error')

    window_errors = results_1_1.plot_squared_td_errors(
        label='1.1',  std=False)
    results_1_2.plot_squared_td_errors(
        label='1.2',  std=False)
    results_1_3.plot_squared_td_errors(
        label='1.3', std=False)
    results_1_4.plot_squared_td_errors(
        label='1.4',  std=False)
    results_1_5.plot_squared_td_errors(
        label='1.5',  std=False)
    results_1_6.plot_squared_td_errors(
        label='1.6',  std=False)
    results_1_7.plot_squared_td_errors(
        label='1.7',  std=False)
    results_1_8.plot_squared_td_errors(
        label='1.8',  std=False)
    results_1_9.plot_squared_td_errors(
        label='1.9',  std=False)
    results_1_10.plot_squared_td_errors(
        label='1.10',  std=False)
    results_1_11.plot_squared_td_errors(
        label='1.11',  std=False)
    results_1_12.plot_squared_td_errors(
        label='1.12',  std=False)
    plt.title('Q-Learning TD-Error Window =' +
              str(window_errors))
    plt.legend()
    plt.show()
    print("AVG Q_Learning - Last 50 Episodes")
    print("1.1 Last 50 episodes:",
          results_1_1.get_mean_step_reward_last_n_episodes())
    print("1.2 Last 50 episodes:",
          results_1_2.get_mean_step_reward_last_n_episodes())
    print("1.3 Last 50 episodes:",
          results_1_3.get_mean_step_reward_last_n_episodes())
    print("1.4 Last 50 episodes:",
          results_1_4.get_mean_step_reward_last_n_episodes())
    print("1.5 Last 50 episodes:",
          results_1_5.get_mean_step_reward_last_n_episodes())
    print("1.6 Last 50 episodes:",
          results_1_6.get_mean_step_reward_last_n_episodes())
    print("1.7 Last 50 episodes:",
          results_1_7.get_mean_step_reward_last_n_episodes())
    print("1.8 Last 50 episodes:",
          results_1_8.get_mean_step_reward_last_n_episodes())
    print("1.9 Last 50 episodes:",
          results_1_9.get_mean_step_reward_last_n_episodes())
    print("1.10 Last 50 episodes:",
          results_1_10.get_mean_step_reward_last_n_episodes())
    print("1.11 Last 50 episodes:",
          results_1_11.get_mean_step_reward_last_n_episodes())
    print("1.12 Last 50 episodes:",
          results_1_12.get_mean_step_reward_last_n_episodes())


def compare_parameters_sarsa(num_episodes, random_seed, config, eps_decay_factor):
    results_1_1 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.9, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_2 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.99, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_3 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.8, epsileps_decay_factoron=0.999, random_seed=random_seed)
    results_1_4 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.7, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_5 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.3, gamma=0.9, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_6 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.4, gamma=0.9, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_7 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.6, gamma=0.9, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_8 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.7, gamma=0.9, eps_decay_factor=0.999, random_seed=random_seed)
    results_1_9 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.9, eps_decay_factor=0.9995, random_seed=random_seed)
    results_1_10 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.9, eps_decay_factor=0.9985, random_seed=random_seed)
    results_1_11 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.9, eps_decay_factor=0.9999, random_seed=random_seed)
    results_1_12 = run_sarsa_agent(WarehouseEnv(
        config), num_episodes=num_episodes, alpha=0.5, gamma=0.9, eps_decay_factor=0.9980, random_seed=random_seed)

    plt.xlabel('Epochen')
    plt.ylabel('∅-Reward pro Step')
    window_rewards = results_1_1.plot_episode_rewards(
        label='1.1', window=int(num_episodes/100))
    results_1_2.plot_episode_rewards(label='1.2', window=int(num_episodes/100))
    results_1_3.plot_episode_rewards(label='1.3', window=int(num_episodes/100))
    results_1_4.plot_episode_rewards(label='1.4', window=int(num_episodes/100))
    results_1_5.plot_episode_rewards(label='1.5', window=int(num_episodes/100))
    results_1_6.plot_episode_rewards(label='1.6', window=int(num_episodes/100))
    results_1_7.plot_episode_rewards(label='1.7', window=int(num_episodes/100))
    results_1_8.plot_episode_rewards(label='1.8', window=int(num_episodes/100))
    results_1_9.plot_episode_rewards(label='1.9', window=int(num_episodes/100))
    results_1_10.plot_episode_rewards(
        label='1.10', window=int(num_episodes/100))
    results_1_11.plot_episode_rewards(
        label='1.11', window=int(num_episodes/100))
    results_1_12.plot_episode_rewards(
        label='1.12', window=int(num_episodes/100))
    plt.title('Sarsa Rewards Window=' +
              str(window_rewards))
    plt.legend()
    plt.show()
    plt.xlabel('Steps')
    plt.ylabel('TD-Error')

    window_errors = results_1_1.plot_squared_td_errors(
        label='1.1',  std=False)
    results_1_2.plot_squared_td_errors(
        label='1.2',  std=False)
    results_1_3.plot_squared_td_errors(
        label='1.3',  std=False)
    results_1_4.plot_squared_td_errors(
        label='1.4',  std=False)
    results_1_5.plot_squared_td_errors(
        label='1.5',  std=False)
    results_1_6.plot_squared_td_errors(
        label='1.6',  std=False)
    results_1_7.plot_squared_td_errors(
        label='1.7',  std=False)
    results_1_8.plot_squared_td_errors(
        label='1.8',  std=False)
    results_1_9.plot_squared_td_errors(
        label='1.9',  std=False)
    results_1_10.plot_squared_td_errors(
        label='1.10',  std=False)
    results_1_11.plot_squared_td_errors(
        label='1.11',  std=False)
    results_1_12.plot_squared_td_errors(
        label='1.12',  std=False)
    plt.title('Sarsa TD-Error Window =' +
              str(window_errors))
    plt.legend()
    plt.show()
    print("AVG Sarsa - Last 50 Episodes")
    print("1.1 Last 50 episodes:",
          results_1_1.get_mean_step_reward_last_n_episodes())
    print("1.2 Last 50 episodes:",
          results_1_2.get_mean_step_reward_last_n_episodes())
    print("1.3 Last 50 episodes:",
          results_1_3.get_mean_step_reward_last_n_episodes())
    print("1.4 Last 50 episodes:",
          results_1_4.get_mean_step_reward_last_n_episodes())
    print("1.5 Last 50 episodes:",
          results_1_5.get_mean_step_reward_last_n_episodes())
    print("1.6 Last 50 episodes:",
          results_1_6.get_mean_step_reward_last_n_episodes())
    print("1.7 Last 50 episodes:",
          results_1_7.get_mean_step_reward_last_n_episodes())
    print("1.8 Last 50 episodes:",
          results_1_8.get_mean_step_reward_last_n_episodes())
    print("1.9 Last 50 episodes:",
          results_1_9.get_mean_step_reward_last_n_episodes())
    print("1.10 Last 50 episodes:",
          results_1_10.get_mean_step_reward_last_n_episodes())
    print("1.11 Last 50 episodes:",
          results_1_11.get_mean_step_reward_last_n_episodes())
    print("1.12 Last 50 episodes:",
          results_1_12.get_mean_step_reward_last_n_episodes())
