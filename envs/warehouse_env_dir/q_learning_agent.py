try:
    from envs.warehouse_env_dir.warehouse_env import WarehouseEnv
    import envs.warehouse_env_dir.plotting
except ModuleNotFoundError:
    from warehouse_env import WarehouseEnv
    import plotting
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools
import matplotlib.style

# Q Function

'''
def q_function(count=50000, steps=1000, alpha=0.1, gamma=1.0, eps=1.0, seed=None):

    def max_action(Q, state, actions):
        values = np.array([Q[state, a] for a in actions])
        action = np.argmax(values)
        # print('best action is', action, actions[action])
        return actions[action]

    config = EnvConfig(seed, 2, 2, 3, steps)
    env = WarehouseEnv(config)

    ALPHA = alpha  # learningrate
    GAMMA = gamma
    EPS = eps  # eps greedy action selection

    Q = {}
    for state in env.get_possible_states():
        for action in env.actions.actions:
            Q[state, action] = 0

    num_ep = count
    total_rewards = np.zeros(num_ep)

    for i in range(num_ep):
        # Print all n episodes
        if i % 1000 == 0:
            print('starting episode', i)
        done = False
        observation = env.reset()
        while not done:
            rand = random.random()
            action = max_action(Q, observation, env.actions.actions) if rand < (1-EPS) \
                else env.actions.get_random_action()
            observation_, reward, game_over, debug = env.step(action)

            action_ = max_action(Q, observation_, env.actions.actions)
            # Update Q table
            Q[observation, action] = Q[observation, action] + ALPHA*(reward +
                                                                     GAMMA*Q[observation_, action_] - Q[observation, action])
            observation = observation_

            if env.game_over:
                break
        # linear decrease of epsilon
        if EPS - 2 / num_ep > 0:
            EPS -= 2 / num_ep
        else:
            EPS = 0
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


def q_function_extended_order(count=50000, steps=1000, alpha=0.1, gamma=1.0, eps=1.0, seed=None, steps_to_request=3, simple_state=True):

    def max_action(Q, state, actions):
        values = np.array([Q[state, a] for a in actions])
        action = np.argmax(values)
        # print('best action is', action, actions[action])
        return actions[action]

    config = EnvConfig(seed, 2, 2, 3, steps,
                       steps_to_request=steps_to_request, simple_state=simple_state)
    env = WarehouseEnv(config)

    ALPHA = alpha  # learningrate
    GAMMA = gamma
    EPS = eps  # eps greedy action selection

    Q = {}
    for state in env.get_possible_states():
        for action in env.actions.actions_extended:
            Q[state, action] = 0

    num_ep = count
    total_rewards = np.zeros(num_ep)

    for i in range(num_ep):
        # Print all n episodes
        if i % 1000 == 0:
            print('starting episode', i)
        done = False
        observation = env.reset()
        while not done:
            # TODO seed?
            rand = random.random()
            action = max_action(Q, observation, env.actions.actions_extended) if rand < (1-EPS) \
                else env.actions.get_random_action_extended()
            observation_, reward, game_over, debug = env.step(action)

            action_ = max_action(Q, observation_, env.actions.actions_extended)
            # Update Q table
            Q[observation, action] = Q[observation, action] + ALPHA*(reward +
                                                                     GAMMA*Q[observation_, action_] - Q[observation, action])
            observation = observation_

            if env.game_over:
                break

        if(True):
            # linear decrease of epsilon
            if EPS - 2 / num_ep > 0:
                EPS -= 2 / num_ep
            else:
                EPS = 0
        else:
            # take random actions until half eppisodes are over
            if i >= num_ep/2 and EPS > 0:
                EPS = 0

        if i % 1000 == 0:
            print('Episode ', i, ' reward is:',
                  env.rewards.get_total_episode_reward())

    env.rewards.set_q(Q)
    # print(Q)
    env.rewards.print_final_reward_infos()
    # env.rewards.plot_total_episode_rewards()
    # env.rewards.plot_episode_rewards(1)
    # env.rewards.plot_episode_rewards(num_ep-1)
    return env.rewards


def q_function_with_idle(count=50000, steps=1000, alpha=0.1, gamma=1.0, eps=1.0, seed=None, steps_to_request=3):

    def max_action(Q, state, actions):
        values = np.array([Q[state, a] for a in actions])
        action = np.argmax(values)
        # print('best action is', action, actions[action])
        return actions[action]

    config = EnvConfig(seed, 2, 2, 3, steps,
                       steps_to_request=steps_to_request)
    env = WarehouseEnv(config)

    ALPHA = alpha  # learningrate
    GAMMA = gamma
    EPS = eps  # eps greedy action selection

    Q = {}
    for state in env.get_possible_states():
        # TODO add actions with articles
        for action in env.actions.actions_with_idle:
            Q[state, action] = 0

    num_ep = count
    total_rewards = np.zeros(num_ep)

    for i in range(num_ep):
        # Print all n episodes
        if i % 1000 == 0:
            print('starting episode', i)
        done = False
        observation = env.reset()
        while not done:
            rand = random.random()
            action = max_action(Q, observation, env.actions.actions_with_idle) if rand < (1-EPS) \
                else env.actions.get_random_action_with_idle()
            observation_, reward, game_over, debug = env.step(action)

            action_ = max_action(
                Q, observation_, env.actions.actions_with_idle)
            # Update Q table
            Q[observation, action] = Q[observation, action] + ALPHA*(reward +
                                                                     GAMMA*Q[observation_, action_] - Q[observation, action])
            observation = observation_

            if env.game_over:
                break

        # linear decrease of epsilon
        if EPS - 2 / num_ep > 0:
            EPS -= 2 / num_ep
        else:
            EPS = 0
        if i % 1000 == 0:
            print('Episode ', i, ' reward is:',
                  env.rewards.get_total_episode_reward())

    env.rewards.set_q(Q)
    # print_q_matrix(Q)
    env.rewards.print_final_reward_infos()
    # env.rewards.plot_total_episode_rewards()
    # env.rewards.plot_episode_rewards(1)
    # env.rewards.plot_episode_rewards(num_ep-1)
    # return env.rewards.all_episode_rewards_per_step
    return env.rewards
'''
# TODO Remove old versions


def q_learning_agent(config, count=100000,  alpha=0.1, gamma=1.0, eps=1.0, linear_eps=True):
    def max_action(Q, state, actions):
        values = np.array([Q[state, a] for a in actions])
        action = np.argmax(values)
        # print('best action is', action, actions[action])
        return actions[action]

    def get_unreached_count(Q):
        count = 0
        for k in Q.keys():
            if Q[k] == 0:
                count += 1
        return count

    env = WarehouseEnv(config)

    ALPHA = alpha  # learningrate
    GAMMA = gamma
    EPS = eps  # eps greedy action selection

    Q = {}
    for state in env.get_possible_states():
        for action in env.actions.actions_extended:
            Q[state, action] = 0

    num_ep = count
    total_rewards = np.zeros(num_ep)
    reached_states_actions = defaultdict(list)
    td_deltas = []
    count_unreached = []

    reached_states_actions = {}
    for state in env.get_possible_states():
        for action in env.actions.actions_extended:
            reached_states_actions[state, action] = 0

    for i in range(num_ep):
        # Print all n episodes
        if i % 1000 == 0:
            print('starting episode', i)
        done = False
        observation = env.reset()
        while not done:
            # TODO seed?
            rand = random.random()
            action = max_action(Q, observation, env.actions.actions_extended) if rand < (1-EPS) \
                else env.actions.get_random_action_extended()
            observation_, reward, game_over, debug = env.step(action)
            if reached_states_actions[observation, action]:
                reached_states_actions[observation, action] += 1
            else:
                reached_states_actions[observation, action] = 1
            action_ = max_action(Q, observation_, env.actions.actions_extended)
            # Update Q table
            td_delta = reward + GAMMA * \
                Q[observation_, action_] - Q[observation, action]
            td_deltas.append(td_delta)

            Q[observation, action] = Q[observation, action] + ALPHA*(reward +
                                                                     GAMMA*Q[observation_, action_] - Q[observation, action])
            observation = observation_

            if env.game_over:
                break

        if(linear_eps):
            # linear decrease of epsilon
            if EPS - 2 / num_ep > 0:
                EPS -= 2 / num_ep
            else:
                EPS = 0
        else:
            # take random actions until half eppisodes are over
            if i >= num_ep/2 and EPS > 0:
                EPS = 0

        count_unreached.append(get_unreached_count(Q))

        if i % 1000 == 0:
            print('Episode ', i, ' reward is:',
                  env.rewards.get_total_episode_reward())

    count_list = []
    unreached_pairs = []
    for v in reached_states_actions:
        if reached_states_actions[v] == 0:
            unreached_pairs.append(v)
        count_list.append(reached_states_actions[v])
    num_bins = [1, 100, 1000, 10000]

    n, bins, patches = plt.hist(x=count_list,
                                bins=num_bins, density=True)
    # print(n, bins, patches)
    # print(unreached_pairs)
    # print(len(count_list), '0:', count_list.count(0), '1:', count_list.count(1))
    # plt.show()
    # plt.bar(['100', '1000', '10000'], n)
    # plt.show()
    # plt.plot(td_deltas)
    # plt.plot(count_unreached)
    # print(min(count_unreached))
    # plt.show()
    env.rewards.set_q(Q)
    # print(Q)
    env.rewards.print_final_reward_infos()
    # env.rewards.plot_total_episode_rewards()
    # env.rewards.plot_episode_rewards(1)
    # env.rewards.plot_episode_rewards(num_ep-1)
    return env.rewards

# TODO remove
# matplotlib.style.use('ggplot')


def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.

    Returns a function that takes the state
    as an input and returns the probabilities
    for each action in the form of a numpy array
    of length of the action space(set of possible actions).
    """
    def policyFunction(state):

        Action_probabilities = np.ones(num_actions,
                                       dtype=float) * epsilon / num_actions

        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities

    return policyFunction


def qLearning(config, num_episodes, discount_factor=1.0,
              alpha=0.6, epsilon=0.5):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""
    env = WarehouseEnv(config)
    num_actions = len(env.actions.actions_extended)
    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(num_actions))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, num_actions)
    td_deltas = []
    q_values_count = []

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset()

        for t in itertools.count():

            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(
                len(action_probabilities)),
                p=action_probabilities)

            # take action and get reward, transit to next state
            next_state, reward, done, _ = env.step(
                env.actions.actions_extended[action])

            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * \
                Q[next_state][best_next_action]
            td_delta = td_target - \
                Q[state][action]
            td_deltas.append(td_delta)
            Q[state][action] += alpha * td_delta
            q_values_count.append(len(Q.items()))

            # done is True if episode terminated
            if done:
                break

            state = next_state

        if ith_episode % 1000 == 0:
            print('Episode ', ith_episode, ' reward is:',
                  env.rewards.get_total_episode_reward())

    plt.plot(td_deltas)
    plt.plot(q_values_count)
    plt.show()

    plotting.plot_episode_stats(stats)
    return env.rewards


def sarsa(config, num_episodes, discount_factor=1.0,
          alpha=0.6, epsilon=0.5):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""
    env = WarehouseEnv(config)
    num_actions = len(env.actions.actions_extended)
    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(num_actions))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, num_actions)
    td_deltas = []
    q_values_count = []

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset()

        for t in itertools.count():

            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(
                len(action_probabilities)),
                p=action_probabilities)

            # take action and get reward, transit to next state
            next_state, reward, done, _ = env.step(
                env.actions.actions_extended[action])

            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * \
                Q[next_state][best_next_action]
            td_delta = td_target - \
                Q[state][action]
            td_deltas.append(td_delta)
            Q[state][action] += alpha * td_delta
            q_values_count.append(len(Q.items()))

            # done is True if episode terminated
            if done:
                break

            state = next_state

        if ith_episode % 1000 == 0:
            print('Episode ', ith_episode, ' reward is:',
                  env.rewards.get_total_episode_reward())

    plt.plot(td_deltas)
    plt.plot(q_values_count)
    plt.show()

    plotting.plot_episode_stats(stats)
    return env.rewards


def qLearning_continuous(config, num_episodes, discount_factor=1.0,
                         alpha=0.1, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""
    env = WarehouseEnv(config)
    num_actions = len(env.actions.actions_extended)
    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(num_actions))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, num_actions)
    td_deltas = []
    q_values_count = []
    rewards = []

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset()

        for t in itertools.count():
            if t % 100000 == 0:
                print('Episode ', t)

            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(
                len(action_probabilities)),
                p=action_probabilities)

            # take action and get reward, transit to next state
            next_state, reward, done, _ = env.step(
                env.actions.actions_extended[action])

            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * \
                Q[next_state][best_next_action]
            td_delta = td_target - \
                Q[state][action]
            td_deltas.append(td_delta)
            rewards.append(reward)
            env.rewards.add_continous_step(reward)
            Q[state][action] += alpha * td_delta
            q_values_count.append(len(Q.items()))

            # done is True if episode terminated
            if done:
                break

            state = next_state

        if ith_episode % 1000 == 0:
            print('Episode ', ith_episode, ' reward is:',
                  env.rewards.get_total_episode_reward())

    plt.plot(td_deltas)
    plt.plot(q_values_count)
    plt.show()
    plt.plot(smoothList(rewards))
    plt.plot(env.rewards.get_smothed_continous_steps())
    plt.show()
    return env.rewards


def smoothList(list, x=100):
    if(len(list) > x):
        ratio = int(len(list)/x)
        smoothed = [0]*(x)
        for i in range(len(smoothed)):
            pos = i*ratio
            smoothed[i] = sum(list[pos:pos+ratio])/float(ratio)
        return smoothed
    return list
