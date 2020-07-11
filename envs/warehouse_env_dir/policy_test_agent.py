import random
import math


class PolicyTestAgent:

    def __init__(self, actions, Q, random_seed=0):
        """
        The Q-values will be stored in a dictionary. Each key will be of the format: ((x, y), a). 
        params:
            actions (list): A list of all the possible action values.
        """
        self.Q = Q

        self.actions = actions

        random.seed(random_seed)

    def get_Q_value(self, state, action):
        """
        Get q value for a state action pair.
        params:
            state (tuple): (x, y) coords in the grid
            action (int): an integer for the action
        """
        return self.Q.get((state, action), 0.0)  # Return 0.0 if state-action pair does not exist

    def act(self, state):
        # Choose the greedy action
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


def run_policy_test_agent(env, num_episodes, random_seed, Q):
    print('Policy Test')
    print('===================================================================================================================')
    episode_scores = []

    actions = env.actions.actions_extended

    agent = PolicyTestAgent(actions=actions, random_seed=random_seed, Q=Q)

    # Storing the path taken and score for the best episode
    best_score = -math.inf
    best_path_actions = list()
    best_score_episodes_taken = 0

    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        episode_score = 0
        episode_actions = []
        while True:
            action = agent.act(state)
            new_state, reward, done, debug = env.step(action)

            episode_score += reward

            state = new_state
            episode_actions.append(action)
            if done:
                break

        episode_scores.append(episode_score)
        # For best episode data
        if episode_score > best_score:
            best_score = episode_score
            best_path_actions = episode_actions
            best_score_episodes_taken = i_episode

        print(
            f'\rEpisode: {i_episode}/{num_episodes}, score: {episode_score}, Average(last 20): {sum(episode_scores[:-20])/len(episode_scores)}', end='')

    print(
        f'\nAfter {num_episodes}, average score: {sum(episode_scores)/len(episode_scores)}, Average(last 20): {sum(episode_scores[:-20])/len(episode_scores)}')
    print(
        f'Best score: {best_score}, Sequence of actions: {[action for action in best_path_actions]}, Reached in {best_score_episodes_taken} episodes')
    print('===================================================================================================================')
    return env.rewards
