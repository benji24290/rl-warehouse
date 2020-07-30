try:
    from envs.warehouse_env_dir.warehouse_env import WarehouseEnv
except ModuleNotFoundError:
    from warehouse_env import WarehouseEnv
import numpy as np
import random

# Random


def random_agent(config, count=100):
    # env = WarehouseEnv(None, 2, 2, 3, 50)

    env = WarehouseEnv(config)

    num_ep = count

    for i in range(num_ep):
        # Print all n episodes
        if i % 1000 == 0:
            print('starting episode', i)
        done = False
        # observation = env.reset()
        env.reset()
        while not done:
            # rand = np.random.random()
            # action = max_action(Q, observation, env.possible_actions)
            [state, reward, game_over, debug] = env.step()
            if env.game_over:
                break
        if i % 1000 == 0:
            print('Episode ', i, ' reward is:',
                  env.rewards.get_total_episode_reward())

    env.rewards.print_final_reward_infos()
    # env.rewards.plot_total_episode_rewards()
    # env.rewards.plot_episode_rewards(1)

    # env.rewards.plot_loot_storage(1)
    # env.rewards.plot_loot_request_updates(1)
    # env.rewards.plot_loot_arrival(1)
    # env.rewards.plot_action_deliver(1)
    # env.rewards.plot_action_order(1)
    # env.rewards.plot_action_store(1)
    return env.rewards
