import random
import math


class SarsaAgent:

    def __init__(self, actions, alpha=0.4, gamma=0.9, random_seed=0):
        """
        The Q-values will be stored in a dictionary. Each key will be of the format: ((x, y), a). 
        params:
            actions (list): A list of all the possible action values.
            alpha (float): step size
            gamma (float): discount factor
        """
        self.Q = {}

        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma

        random.seed(random_seed)

    def get_Q_value(self, state, action):
        """
        Get q value for a state action pair.
        params:
            state (tuple): (x, y) coords in the grid
            action (int): an integer for the action
        """
        return self.Q.get((state, action), 0.0)  # Return 0.0 if state-action pair does not exist

    def act(self, state, epsilon=0.1):
        # Choose a random action
        if random.random() < epsilon:
            action = random.choice(self.actions)
        # Choose the greedy action
        else:
            # Get all the Q-values for all possible actions for the state
            q_values = [self.get_Q_value(state, action)
                        for action in self.actions]
            maxQ = max(q_values)
            # There might be cases where there are multiple actions with the same high q_value. Choose randomly then
            count_maxQ = q_values.count(maxQ)
            if count_maxQ > 1:
                # Get all the actions with the maxQ
                best_action_indexes = [i for i in range(
                    len(self.actions)) if q_values[i] == maxQ]
                action_index = random.choice(best_action_indexes)
            else:
                action_index = q_values.index(maxQ)

            action = self.actions[action_index]

        return action

    def learn(self, state, action, reward, next_state, next_state_action):
        """
        Rethink the learn portion to account for first time encounters of state and terminal conditions.
        """
        td_error = 0
        visited = len(self.Q)
        q_next = self.get_Q_value(next_state, next_state_action)

        # If this is the first time the state action pair is encountered
        q_current = self.Q.get((state, action), None)
        if q_current is None:
            self.Q[(state, action)] = reward
            td_error = reward
        else:
            self.Q[(state, action)] = q_current + (self.alpha *
                                                   (reward + self.gamma * q_next - q_current))
            td_error = reward + self.gamma * q_next - q_current
        return td_error, visited


def run_sarsa_agent(env, num_episodes, alpha,
                    gamma, eps_decay_factor, random_seed):
    # <------------Sarsa ---------------------->
    print('Stats for Sarsa')
    print('===================================================================================================================')
    episode_scores = []
    epsilon = 1
    eps_min = 0.05

    actions = env.actions.actions_extended

    agent = SarsaAgent(actions=actions, alpha=alpha,
                       gamma=alpha, random_seed=random_seed)

    # Storing the path taken and score for the best episode
    best_score = -math.inf
    best_path_actions = list()
    best_score_episodes_taken = 0

    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        episode_score = 0
        episode_actions = []
        while True:
            action = agent.act(state, epsilon=epsilon)
            new_state, reward, done, debug = env.step(action)

            episode_score += reward

            new_state_action = agent.act(new_state)
            td_error, visited = agent.learn(state, action, reward,
                                            new_state, new_state_action)

            env.rewards.squared_td_errors.append(td_error*td_error)
            env.rewards.visited_states.append(visited)
            env.rewards.epsilons.append(epsilon)

            state = new_state
            episode_actions.append(action)
            if done:
                break

        episode_scores.append(episode_score)
        # Decay epsilon
        epsilon = max(epsilon * eps_decay_factor, eps_min)

        # For best episode data
        if episode_score > best_score:
            best_score = episode_score
            best_path_actions = episode_actions
            best_score_episodes_taken = i_episode

        print(
            f'\rEpisode: {i_episode}/{num_episodes}, score: {episode_score}, Average(last 20): {sum(episode_scores[:-20])/len(episode_scores)}, epsilon: {epsilon}', end='')

    print(
        f'\nAfter {num_episodes}, average score: {sum(episode_scores)/len(episode_scores)}, Average(last 20): {sum(episode_scores[:-20])/len(episode_scores)}')
    print(
        f'Best score: {best_score}, Sequence of actions: {[action for action in best_path_actions]}, Reached in {best_score_episodes_taken} episodes')
    print('===================================================================================================================')
    env.rewards.q = agent.Q
    return env.rewards
