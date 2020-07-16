try:
    from envs.warehouse_env_dir.warehouse_env import WarehouseEnv
except ModuleNotFoundError:
    from warehouse_env import WarehouseEnv
import numpy as np
import random

# Heuristic


def heuristic(config, count=1000, version='v1'):
    # TODO Remove simple state, useless
    # env = WarehouseEnv(None, 2, 2, 3, 50)
    if(version == 'v1'):
        check_for_orders = False
        can_idle = False
    elif(version == 'v2'):
        check_for_orders = True
        can_idle = False
    elif(version == 'v3'):
        check_for_orders = False
        can_idle = True
    elif(version == 'v4'):
        check_for_orders = True
        can_idle = True
    else:
        raise Exception(version, 'is not a valid version')

    env = WarehouseEnv(config)
    num_ep = count

    Q = {}
    for state in env.get_possible_states():
        # TODO add actions with articles
        for action in env.actions.actions:
            Q[state, action] = 0

    for i in range(num_ep):
        # Print all n episodes
        if i % 1000 == 0:
            print('starting episode', i)
        done = False
        # observation = env.reset()
        env.reset()
        while not done:
            action = None
            a_id = None
            # prio 1: fullfill requests if article is in store
            for req in env.requests.requests:
                for space in env.storage.storage_spaces:
                    if req.article == space.article and action == None:
                        # print('deliver:', req.article.name, space.article.name)
                        action = 'DELIVER'
                        a_id = req.article.id
            # prio 2: store articles from arrival
            if(action == None):
                if len(env.storage.get_possible_spaces()) > 0:
                    for arr_space in env.arrivals.arrivals:
                        if arr_space.article:
                            action = 'STORE'
                            a_id = arr_space.article.id

            # prio 3: order article from wich the least are stored
            if(action == None):
                # TODO least stored
                for possible in env.possible_articles.articles:
                    empty_spaces_storage_arrival = env.storage.get_simple_storage_state().count(
                        0)+env.arrivals.get_simple_arrivals_state().count(0)
                    if (possible.id not in env.storage.get_simple_storage_state() and empty_spaces_storage_arrival > 0):

                        if(check_for_orders):
                            if(empty_spaces_storage_arrival-len(env.orders.orders) > 0):
                                if len(env.orders.orders) < 1:
                                    action = 'ORDER_'+str(possible.id)
                                    a_id = possible.id
                                else:
                                    for order in env.orders.orders:
                                        if(order.article.id is not possible.id):
                                            # print(possible.id, 'is not in:',
                                            #      env.storage.get_simple_storage_state())
                                            action = 'ORDER_'+str(possible.id)
                                            a_id = possible.id
                                        else:
                                            # print(possible.id, 'was already ordered',
                                            #      order.article.id)
                                            pass
                            else:
                                # print('empty=', empty_spaces_storage_arrival,
                                #     'open orders=', len(env.orders.orders))
                                pass
                        else:
                            action = 'ORDER_'+str(possible.id)
                            a_id = possible.id
            # prio 4: do nothing
            if(can_idle):
                if(action == None):
                    action = 'IDLE'
            else:
                action = random.choice(env.actions.actions)
            Q[env.get_state(), action] = 100
            env.rewards.all_actions.append(action)
            [state, reward, game_over, debug] = env.step(action, a_id)
            if env.game_over:
                break
        if i % 1000 == 0:
            print('Episode ', i, ' reward is:',
                  env.rewards.get_total_episode_reward())

    env.rewards.set_q(Q)
    # print_q_matrix(Q)
    env.rewards.print_final_reward_infos()
    # env.rewards.plot_total_episode_rewards()
    # env.rewards.plot_episode_rewards(1)
    # env.rewards.plot_episode_rewards(num_ep-1)
    return env.rewards
