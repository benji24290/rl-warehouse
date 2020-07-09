

try:
    from envs.warehouse_env_dir.warehouse_env import WarehouseEnv
    from envs.warehouse_env_dir.env_config import EnvConfig
    from envs.warehouse_env_dir.q_learning_agent import q_learning_agent
    from envs.warehouse_env_dir.q_learning_agent import qLearning
    from envs.warehouse_env_dir.q_learning_agent import qLearning_continuous
    from envs.warehouse_env_dir.heuristic_agent import heuristic
    from envs.warehouse_env_dir.random_agent import random_agent
except ModuleNotFoundError:
    from env_config import EnvConfig
    from q_learning_agent import q_learning_agent
    from q_learning_agent import qLearning
    from q_learning_agent import qLearning_continuous
    from heuristic_agent import heuristic
    from random_agent import random_agent
    from warehouse_env import WarehouseEnv
import matplotlib.pyplot as plt


if __name__ == '__main__':
    def tests():

        # Test Summary 1 One-Art---3Steps to request
        if(False):
            rew_q_e_order = q_learning_agent(seed=1234,  turns=100,
                                             steps_to_request=4)
            rew_h_v4 = heuristic(10000, 100, version='v4',
                                 seed=1234,  steps_to_request=3)
            plt.xlabel('Epochen')
            plt.ylabel('∅-Reward pro Step')
            plt.title('Bestellung alle 4 Steps')
            plt.plot(rew_q_e_order.get_smooth_all_episode_rewards_per_step(),
                     label='q-func-v2')
            plt.plot(
                rew_h_v4.get_smooth_all_episode_rewards_per_step(), label='heur')
            plt.legend()
            plt.show()

            rew_q_e_order.plot_pos_neg_rewards(name='q-ext-a0.1-order')
            rew_q_e_order.plot_episode_rewards(9800, name='Q-Function V2')
            rew_h_v4.plot_episode_rewards(9800, name='Heuristik')

        if(False):
            config = EnvConfig(seed=1234,  turns=100,
                               steps_to_request=4)
            rew_q_e = q_learning_agent(
                config, alpha=0.1, count=1000, linear_eps=True)
            rew_q_new = qLearning(
                config, 50000)
            rew_q_new_05 = qLearning(
                config, 50000, discount_factor=0.5)
            rew_rand = random_agent(config)
            rew_h_v3 = heuristic(config, count=1000, version='v3')
            rew_h_v4 = heuristic(config, count=1000, version='v4')
            plt.xlabel('Epochen')
            plt.ylabel('∅-Reward pro Step')
            plt.title('Bestellung alle 4 Steps')
            plt.plot(rew_q_e.get_smooth_all_episode_rewards_per_step(),
                     label='q-func')
            plt.plot(rew_q_new.get_smooth_all_episode_rewards_per_step(),
                     label='q-func-new')
            plt.plot(rew_q_new_05.get_smooth_all_episode_rewards_per_step(),
                     label='q-func-new_05')
            plt.plot(
                rew_rand.get_smooth_all_episode_rewards_per_step(), label='rand')
            plt.plot(
                rew_h_v3.get_smooth_all_episode_rewards_per_step(), label='heur-v3')
            plt.plot(
                rew_h_v4.get_smooth_all_episode_rewards_per_step(), label='heur-v4')
            plt.legend()
            plt.show()

            # rew_q_e.print_q_matrix()
            # rew_q_e_order.print_q_matrix()
            # rew_q_e_order.print_q_matrix()

            rew_q_e.plot_pos_neg_rewards(name='q-ext-a0.1')
            rew_q_new.plot_pos_neg_rewards(name='q-ext-a0.1')
            rew_q_e.plot_episode_rewards(980, name='Q-Function')
            rew_q_new.plot_episode_rewards(4980, name='Q-Function-new')
            rew_h_v3.plot_episode_rewards(980, name='Heuristik')
            rew_h_v4.plot_episode_rewards(980, name='Heuristik')

        if(True):
            config = EnvConfig(seed=1234,  turns=100,
                               steps_to_request=4)
            config_cont = EnvConfig(seed=1234,  turns=200000,
                                    steps_to_request=4)
            rew_q_e = q_learning_agent(
                config, alpha=0.1, count=1000, linear_eps=True)
            rew_q_new = qLearning_continuous(
                config_cont, 1)
            rew_q_new_05 = qLearning_continuous(
                config_cont, 1, discount_factor=0.5)
            rew_rand = random_agent(config)
            rew_h_v3 = heuristic(config, count=1000, version='v3')
            rew_h_v4 = heuristic(config, count=1000, version='v4')

            plt.xlabel('Epochen')
            plt.ylabel('∅-Reward pro Step')
            plt.title('Bestellung alle 4 Steps')
            plt.plot(rew_q_e.get_smooth_all_episode_rewards_per_step(),
                     label='q-func')
            plt.plot(rew_q_new.get_smothed_continous_steps(),
                     label='q-func-new')
            plt.plot(rew_q_new_05.get_smothed_continous_steps(),
                     label='q-func-new_05')
            plt.plot(
                rew_rand.get_smooth_all_episode_rewards_per_step(), label='rand')
            plt.plot(
                rew_h_v3.get_smooth_all_episode_rewards_per_step(), label='heur-v3')
            plt.plot(
                rew_h_v4.get_smooth_all_episode_rewards_per_step(), label='heur-v4')
            plt.legend()
            plt.show()

            # rew_q_e.print_q_matrix()
            # rew_q_e_order.print_q_matrix()
            # rew_q_e_order.print_q_matrix()

            rew_q_e.plot_pos_neg_rewards(name='q-ext-a0.1')
            rew_q_new.plot_pos_neg_rewards(name='q-ext-a0.1')
            rew_q_e.plot_episode_rewards(980, name='Q-Function')
            rew_q_new.plot_episode_rewards(1, name='Q-Function-new')
            rew_h_v3.plot_episode_rewards(980, name='Heuristik')
            rew_h_v4.plot_episode_rewards(980, name='Heuristik')
    tests()
