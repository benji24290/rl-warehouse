try:
    from envs.warehouse_env_dir.warehouse_env import WarehouseEnv
    from envs.warehouse_env_dir.env_config import EnvConfig
    from envs.warehouse_env_dir.heuristic_agent import heuristic
    from envs.warehouse_env_dir.q_learning_agent import run_q_learning_agent
    from env.warehouse_env_dir.sarsa_agent import run_sarsa_agent
    from env.warehouse_env_dir.policy_test_agent import run_policy_test_agent
    from env.warehouse_env_dir.compare_parameters import compare_epsilon_decay, compare_alpha, compare_gamma
except ModuleNotFoundError:
    from warehouse_env import WarehouseEnv
    from env_config import EnvConfig
    from heuristic_agent import heuristic
    from q_learning_agent import run_q_learning_agent
    from sarsa_agent import run_sarsa_agent
    from policy_test_agent import run_policy_test_agent
    from compare_parameters import compare_epsilon_decay, compare_alpha, compare_gamma


import matplotlib.pyplot as plt
import math
import numpy as np


eps_decay_factor = 0.999
eps_min = 0.05

num_episodes = 50000
train_episodes = 800000
alpha = 0.5
gamma = 0.9

random_seed = 60
config = EnvConfig(seed=1234,  turns=100)


alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
gammas = [0.9, 0.8, 0.7, 0.6, 0.5]
eps_decay_factors = [0.9985, 0.999, 0.9995, 0.9999, 1]
# eps_decay_factors = [1]
best_epsilon_dec_q = 0.99999374970701
best_alpha_q = 0.9
best_gamma_q = 0.8

best_epsilon_dec_sarsa = 0.99999374970701
best_alpha_sarsa = 0.6
best_gamma_sarsa = 0.6

# To deisable sections of this experiment
evaluate_params = True
train = False
compare_policies = False  # train also needs to be true


if __name__ == '__main__':

    if(evaluate_params):
        # Q-Learning

        compare_epsilon_decay(num_episodes=num_episodes, random_seed=random_seed, gamma=gamma,
                              alpha=alpha, config=config, eps_decay_factors=eps_decay_factors, learner="Q-Learning")
        compare_alpha(num_episodes=num_episodes, random_seed=random_seed,
                      config=config, eps_decay_factor=eps_decay_factor, gamma=gamma, learner="Q-Learning", alphas=alphas)

        compare_gamma(num_episodes=num_episodes, random_seed=random_seed,
                      config=config, eps_decay_factor=eps_decay_factor, alpha=alpha, learner="Q-Learning", gammas=gammas)

        # Sarsa

        compare_epsilon_decay(num_episodes=num_episodes, random_seed=random_seed, gamma=gamma, alpha=alpha,
                              config=config, eps_decay_factors=eps_decay_factors, learner="Sarsa")
        compare_alpha(num_episodes=num_episodes, random_seed=random_seed,
                      config=config, eps_decay_factor=eps_decay_factor, gamma=gamma, learner="Sarsa", alphas=alphas)
        compare_gamma(num_episodes=num_episodes, random_seed=random_seed,
                      config=config, eps_decay_factor=eps_decay_factor, alpha=alpha, learner="Sarsa", gammas=gammas)

    if(train):
        # train policy with best parameters
        results_sarsa = run_sarsa_agent(WarehouseEnv(
            config), num_episodes=train_episodes, alpha=best_alpha_sarsa, gamma=best_gamma_sarsa, eps_decay_factor=best_epsilon_dec_sarsa, random_seed=random_seed)

        results_q_learning = run_q_learning_agent(WarehouseEnv(
            config), num_episodes=train_episodes, alpha=best_alpha_q, gamma=best_gamma_q, eps_decay_factor=best_epsilon_dec_q, random_seed=random_seed)
        # Plot Rewards
        plt.xlabel('Episoden')
        plt.ylabel('∅-Reward pro Step')
        plt.title('Rewards')
        results_q_learning.plot_episode_rewards(
            label='Q-Learning', std=True)
        results_sarsa.plot_episode_rewards(
            label='Sarsa',  std=True)
        plt.legend()
        plt.show()

        # Plot TD-Error
        plt.xlabel('Steps')
        plt.ylabel('Squared TD-Error')
        # should not be in same graph and should be log
        window_errors = results_sarsa.plot_squared_td_errors(
            label='Sarsa', std=False)
        results_q_learning.plot_squared_td_errors(
            label='Q-Learning', std=False)
        plt.title('TD-Error - Window=' +
                  str(window_errors))
        plt.legend()
        plt.show()

        # Plot visited
        plt.xlabel('Steps')
        plt.ylabel('Besuchte S,A Paare')
        # should not be in same graph and should be log
        window_visited = results_sarsa.plot_visited_s_a(
            label='Sarsa', std=False)
        results_q_learning.plot_visited_s_a(
            label='Q-Learning', std=False)
        plt.title('Besuchte S,A - Window=' +
                  str(window_visited))
        plt.legend()
        plt.show()

        plt.xlabel('Steps')
        plt.ylabel('Epsilon')
        window_epsilon = results_q_learning.plot_epsilons(
            label='Q-Learning',  std=False)
        results_sarsa.plot_epsilons(
            label='Sarsa',  std=False)

        plt.title('Epsilon-Decay - Window=' +
                  str(window_epsilon))
        plt.legend()
        plt.show()

        results_sarsa.plot_pos_neg_rewards(name='Sarsa')
        results_q_learning.plot_pos_neg_rewards(name='Q-Learning')

        if(compare_policies):
            # Test learned Policies
            results_q_learning_policy = run_policy_test_agent(env=WarehouseEnv(
                config), num_episodes=1000, Q=results_q_learning.q, random_seed=random_seed)
            results_sarsa_policy = run_policy_test_agent(env=WarehouseEnv(
                config), num_episodes=1000, Q=results_sarsa.q, random_seed=random_seed)

            rew_h_v4 = heuristic(config, count=1000, version='v4')

            plt.xlabel('Episoden')
            plt.ylabel('∅-Reward pro Step')
            results_q_learning_policy.plot_episode_rewards(
                label='Q-Learning',  std=False)
            results_sarsa_policy.plot_episode_rewards(
                label='Sarsa',  std=False)
            rew_h_v4.plot_episode_rewards(
                label='Heuristik',  std=False)
            plt.legend()
            plt.show()

            results_q_learning_policy.plot_step_rewards_of_episode(
                980)
            results_sarsa_policy.plot_step_rewards_of_episode(980)
            rew_h_v4.plot_step_rewards_of_episode(980)
