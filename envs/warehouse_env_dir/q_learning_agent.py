import random
import math


class QAgent:

    def __init__(self, actions, alpha=0.4, gamma=0.9, random_seed=0):
        self.Q = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        random.seed(random_seed)

    def get_Q_value(self, state, action):
        return self.Q.get((state, action), 0.0)

    def act(self, state, epsilon=0.1):
        # Choose a random action
        if random.random() < epsilon:
            action = random.choice(self.actions)
        # Choose the greedy action
        else:
            action = self.greedy_action_selection(state)

        return action

    def learn(self, state, action, reward, next_state):
        # Update Q-Values
        q_next = self.get_Q_value(
            state=next_state,
            action=self.greedy_action_selection(next_state)
        )
        td_error = 0
        visited = len(self.Q)

        # First visit add Reward as Q-Value
        q_current = self.Q.get((state, action), None)
        if q_current is None:
            self.Q[(state, action)] = reward
            td_error = reward
        else:
            self.Q[(state, action)] = q_current + (self.alpha *
                                                   (reward + self.gamma * q_next - q_current))
            td_error = reward + self.gamma * q_next - q_current
        return td_error, visited

    def greedy_action_selection(self, state):
        # Get action with the highest Q-value
        q_values = [self.get_Q_value(state, action) for action in self.actions]
        maxQ = max(q_values)
        count_maxQ = q_values.count(maxQ)
        if count_maxQ > 1:
            best_action_indexes = [i for i in range(
                len(self.actions)) if q_values[i] == maxQ]
            action_index = random.choice(best_action_indexes)
        else:
            action_index = q_values.index(maxQ)

        return self.actions[action_index]


def run_q_learning_agent(env, num_episodes, alpha,
                         gamma, eps_decay_factor, random_seed):
    print('Q Learning')
    print('____________________________________________________________________________')
    episode_scores = []
    epsilon = 1
    eps_min = 0.05
    actions = env.actions.actions_extended

    agent = QAgent(actions=actions, alpha=alpha,
                   gamma=alpha, random_seed=random_seed)

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

            td_error, visited = agent.learn(state, action, reward, new_state)

            env.rewards.squared_td_errors.append(td_error*td_error)
            env.rewards.visited_states.append(visited)
            env.rewards.epsilons.append(epsilon)

            state = new_state
            episode_actions.append(action)
            if done:
                break
        episode_scores.append(episode_score)

        epsilon = max(epsilon * eps_decay_factor, eps_min)

        # best?
        if episode_score > best_score:
            best_score = episode_score
            best_path_actions = episode_actions
            best_score_episodes_taken = i_episode

        print(
            f'\rEpisode: {i_episode}/{num_episodes}, score: {episode_score}, Average score/step(last 50E):  {sum(episode_scores[-50:])/50/100}, epsilon: {epsilon}', end='')

    print(
        f'\nAfter {num_episodes}, average score: {sum(episode_scores)/len(episode_scores)}, Average score/step(last 50E): {sum(episode_scores[-50:])/50/100}')
    print(
        f'Best score: {best_score}, Sequence of actions: {[action for action in best_path_actions]}, Reached in {best_score_episodes_taken} episodes')
    print('________________________________________________________________________________')
    env.rewards.q = agent.Q
    return env.rewards
