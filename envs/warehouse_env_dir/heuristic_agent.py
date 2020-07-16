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
            # prio 4: do nothing
            if(can_idle):
                if(action == None):
                    action = 'IDLE'
            Q[env.get_state(), action] = 100
            #print(env.get_state(), action)
            [state, reward, game_over, debug] = env.step(action)
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


def create_heuristic_policy(config):
    env = WarehouseEnv(config)

    Q = {}

    i = 0
    for state in env.get_possible_states():
        if i % 10000 == 0:
            print('starting episode', i)
        # TODO add actions with articles
        best_action = evaluate_state(state, env)
        for action in env.actions.actions_extended:
            if(action == best_action):
                Q[(state, action)] = 100
            else:
                # print(action, "not", best_action)
                Q[(state, action)] = 0
        i += 1
    env.rewards.q = Q
    env.rewards.export_q("heuristic.csv")


def evaluate_state(state, env):
    storage = state[0:3]
    requests = state[3:5]
    arrivals = state[5:7]
    orders = state[7:9]
    possible_art = env.possible_articles.get_possible_articles()[::-1]
    # prio 1: fullfill requests if article is in store
    action = None
    for req in requests:
        for space in storage:
            if req != 0 and req == space and action == None:
                action = 'DELIVER'
    # prio 2: store articles from arrival
    if(action == None):
        if storage.count(0) > 0:
            for arr_space in arrivals:
                if arr_space > 0:
                    action = 'STORE'
    # prio 3: order article if not in store or ordered
    if(action == None):
                # TODO least stored
        for possible in env.possible_articles.articles:
            empty_spaces_storage_arrival = storage.count(
                0)+arrivals.count(0)
            if (possible.id not in storage and empty_spaces_storage_arrival > 0):
                if(empty_spaces_storage_arrival-2-orders.count(0) > 0):
                    if 2-orders.count(0) < 1:
                        action = 'ORDER_'+str(possible.id)
                    else:
                        for order in orders:
                            if(order is not possible.id):
                                        # print(possible.id, 'is not in:',
                                        #      env.storage.get_simple_storage_state())
                                action = 'ORDER_'+str(possible.id)
    # prio 4: idle
    if(action == None):
        action = 'IDLE'

    return action
    # When 000,00,00,00
    # print(storage, requests, arrivals, orders)


def evaluate_state_old(state, env):
    storage = state[0:3]
    requests = state[3:5]
    arrivals = state[5:7]
    orders = state[7:9]
    possible_art = env.possible_articles.get_possible_articles()[::-1]
    # prio 1: fullfill requests if article is in store
    for art in possible_art:
        if(requests.count(art) > 0 and storage.count(art) > 0):
            return 'DELIVER'
    # prio 2: store articles from arrival
    for arrival in arrivals:
        if(arrival > 0 and storage.count(0) > 0):
            return 'STORE'
    # prio 3: order article if not in store or ordered
    for art in possible_art:
        if(storage.count(art)+orders.count(art) < 1 and (storage.count(0)+arrivals.count(0)-2+orders.count(0)) > 0):
            if(arrivals.count(art) < 1):
                return 'ORDER_'+str(art)
    for art in possible_art:
        if(storage.count(art)-requests.count(art) < 1 and (storage.count(0)+arrivals.count(0)-2+orders.count(0)) > 0):
            if(arrivals.count(art) < 1):
                return 'ORDER_'+str(art)
    # prio 4: idle
    return 'IDLE'

    # When 000,00,00,00
    # print(storage, requests, arrivals, orders)
