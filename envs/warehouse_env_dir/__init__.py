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
train_episodes = 8000
alpha = 0.5
gamma = 0.9

random_seed = 60
config = EnvConfig(seed=1234,  turns=100)


#alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
#gammas = [0.9, 0.8, 0.7, 0.6, 0.5]
alphas = [0.4, 0.3, 0.2, 0.1, 0.01]
gammas = [0.9995, 0.999, 0.995, 0.99, 0.95]
eps_decay_factors = [0.9985, 0.999, 0.9995, 0.9999, 1]
# eps_decay_factors = [1]
best_epsilon_dec_q = 0.99999374970701
best_alpha_q = 0.1
best_gamma_q = 0.95

best_epsilon_dec_sarsa = 0.99999374970701
best_alpha_sarsa = 0.1
best_gamma_sarsa = 0.9

# To deisable sections of this experiment
evaluate_params = True
train = True
compare_policies = True  # train also needs to be true


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

        results_q_learning.plot_exploration_both(
            other_eps=results_sarsa.epsilons, other_visited=results_sarsa.visited_states)

        results_sarsa.plot_pos_neg_rewards(name='Sarsa')
        results_q_learning.plot_pos_neg_rewards(name='Q-Learning')

        results_sarsa.export_q("sarsa.csv")

        results_q_learning.export_q("q-learning.csv")

    if(compare_policies):
        # Test learned Policies
        q_env = WarehouseEnv(config)
        actions = q_env.actions.actions_extended
        episodes = 1000

        results_q_learning_policy = run_policy_test_agent(
            env=q_env, num_episodes=1000, Q=q_env.rewards.import_q("q-learning.csv"), random_seed=random_seed)
        sarsa_env = WarehouseEnv(config)
        results_sarsa_policy = run_policy_test_agent(
            env=sarsa_env, num_episodes=1000, Q=sarsa_env.rewards.import_q("sarsa.csv"), random_seed=random_seed)

        rew_h_v4 = heuristic(config, count=1000, version='v4')

        results_q_learning_policy.print_actions(actions, episodes)
        results_sarsa_policy.print_actions(actions, episodes)
        rew_h_v4.print_actions(actions, episodes)

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

        results_q_learning_policy.plot_state_info_from_episode(
            980, 'Q-Learning ')
        results_sarsa_policy.plot_state_info_from_episode(
            980, 'Sarsa ')
        rew_h_v4.plot_state_info_from_episode(
            980, 'Heuristik ')
