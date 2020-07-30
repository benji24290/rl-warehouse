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


def compare_epsilon_decay(num_episodes, random_seed, config, eps_decay_factors, alpha, gamma, learner):
    run_agent = None
    if(learner == "Q-Learning"):
        run_agent = run_q_learning_agent
    elif(learner == "Sarsa"):
        run_agent = run_sarsa_agent
    else:
        raise Exception("Unknown learner:", learner)
    results = []
    for i in range(len(eps_decay_factors)):
        result = run_agent(WarehouseEnv(
            config), num_episodes=num_episodes, alpha=alpha, gamma=gamma,  eps_decay_factor=eps_decay_factors[i], random_seed=random_seed)
        results.append(result)
    # Episodes
    plt.xlabel('Steps')
    plt.ylabel('Epsilon')
    window_rewards = 0
    for i in range(len(results)):
        if i == 0:
            window_rewards = results[i].plot_epsilons(
                label='E2.'+str(i+1),  std=False)
        else:
            results[i].plot_epsilons(
                label='E2.'+str(i+1), std=False)

    plt.title(learner+' Epsilon-Decay - Window=' +
              str(window_rewards))
    plt.legend()
    plt.show()

    plt.xlabel('Steps')
    plt.ylabel('Besuchte S,A Paare')
    window_errors = 0
    for i in range(len(results)):
        if i == 0:
            window_errors = results[i].plot_visited_s_a(
                label='E2.'+str(i+1),  std=False)
        else:
            results[i].plot_visited_s_a(
                label='E2.'+str(i+1),  std=False)
    plt.title(learner+' Besuchte S,A - Window=' +
              str(window_errors))
    plt.legend()
    plt.show()

    print("AVG "+learner+" - Last 50 Episodes")
    for i in range(len(results)):
        print("E2."+str(i+1)+":",
              results[i].get_mean_step_reward_last_n_episodes())
    print("New States  state n/5 - "+learner)
    for i in range(len(results)):
        print("E2."+str(i+1)+":",
              results[i].visited_states[num_episodes*100] -
              results[i].visited_states[num_episodes*100-int(num_episodes*100/5)], "("+str(results[i].visited_states[num_episodes*100])+")")


def compare_alpha(num_episodes, random_seed, config, eps_decay_factor, alphas, gamma, learner):
    run_agent = None
    if(learner == "Q-Learning"):
        run_agent = run_q_learning_agent
    elif(learner == "Sarsa"):
        run_agent = run_sarsa_agent
    else:
        raise Exception("Unknown learner:", learner)
    results = []
    for i in range(len(alphas)):
        result = run_agent(WarehouseEnv(
            config), num_episodes=num_episodes, alpha=alphas[i], gamma=gamma,  eps_decay_factor=eps_decay_factor, random_seed=random_seed)
        results.append(result)
    # Episodes
    plt.xlabel('Episoden')
    plt.ylabel('∅-Reward pro Step')
    window_rewards = 0
    for i in range(len(results)):
        if i == 0:
            window_rewards = results[i].plot_episode_rewards(
                label='A2.'+str(i+1),  std=False)
        else:
            results[i].plot_episode_rewards(
                label='A2.'+str(i+1),  std=False)

    plt.title(learner+' Rewards - Window=' +
              str(window_rewards))
    plt.legend()
    plt.show()

    plt.xlabel('Steps')
    plt.ylabel('Squared TD-Error')
    window_errors = 0
    for i in range(len(results)):
        if i == 0:
            window_errors = results[i].plot_squared_td_errors(
                label='A2.'+str(i+1),  std=False)
        else:
            results[i].plot_squared_td_errors(
                label='A2.'+str(i+1),  std=False)
    plt.title(learner+' TD-Error - Window=' +
              str(window_errors))
    plt.legend()
    plt.show()

    print("AVG "+learner+" - Last 50 Episodes")
    for i in range(len(results)):
        print("A2."+str(i+1)+":",
              results[i].get_mean_step_reward_last_n_episodes())


def compare_gamma(num_episodes, random_seed, config, eps_decay_factor, alpha, gammas, learner):
    run_agent = None
    if(learner == "Q-Learning"):
        run_agent = run_q_learning_agent
    elif(learner == "Sarsa"):
        run_agent = run_sarsa_agent
    else:
        raise Exception("Unknown learner:", learner)
    results = []
    for i in range(len(gammas)):
        result = run_agent(WarehouseEnv(
            config), num_episodes=num_episodes, alpha=alpha, gamma=gammas[i],  eps_decay_factor=eps_decay_factor, random_seed=random_seed)
        results.append(result)
    # Episodes
    plt.xlabel('Episoden')
    plt.ylabel('∅-Reward pro Step')
    window_rewards = 0
    for i in range(len(results)):
        if i == 0:
            window_rewards = results[i].plot_episode_rewards(
                label='G2.'+str(i+1),  std=False)
        else:
            results[i].plot_episode_rewards(
                label='G2.'+str(i+1),  std=False)

    plt.title(learner+' Rewards - Window=' +
              str(window_rewards))
    plt.legend()
    plt.show()

    plt.xlabel('Steps')
    plt.ylabel('Squared TD-Error')
    window_errors = 0
    for i in range(len(results)):
        if i == 0:
            window_errors = results[i].plot_squared_td_errors(
                label='G2.'+str(i+1),  std=False)
        else:
            results[i].plot_squared_td_errors(
                label='G2.'+str(i+1),  std=False)
    plt.title(learner+' TD-Error - Window=' +
              str(window_errors))
    plt.legend()
    plt.show()

    print("AVG "+learner+" - Last 50 Episodes")
    for i in range(len(results)):
        print("G2."+str(i+1)+":",
              results[i].get_mean_step_reward_last_n_episodes())
