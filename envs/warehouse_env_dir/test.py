try:
    from envs.warehouse_env_dir.warehouse_env import WarehouseEnv
    from envs.warehouse_env_dir.env_config import EnvConfig
    from envs.warehouse_env_dir.heuristic_agent import heuristic
    from envs.warehouse_env_dir.q_learning_agent2 import QAgent
    from env.warehouse_env_dir.sarsa_agent import SarsaAgent
except ModuleNotFoundError:
    from warehouse_env import WarehouseEnv
    from env_config import EnvConfig
    from heuristic_agent import heuristic
    from q_learning_agent2 import QAgent
    from sarsa_agent import SarsaAgent
    import plotting


import matplotlib.pyplot as plt
import math

q_episode_scores = []
sarsa_episode_scores = []
expected_sarsa_episode_scores = []

# Epsilon greedy action selection
eps_decay_factor = 0.9999  # After every episode, eps is 0.9 times the previous one
eps_min = 0.05  # 10% exploration is compulsory till the end

num_episodes = 50000
alpha = 0.6
gamma = 0.999

random_seed = 60
config = EnvConfig(seed=1234,  turns=100,
                   steps_to_request=4)
# <------------Sarsa ---------------------->
print('Stats for Sarsa')
print('===================================================================================================================')
epsilon = 1

gridworld = WarehouseEnv(config)
actions = gridworld.actions.actions_extended

agent = SarsaAgent(actions=actions, alpha=alpha,
                   gamma=alpha, random_seed=random_seed)

# Storing the path taken and score for the best episode
best_score = -math.inf
best_path_actions = list()
best_score_episodes_taken = 0

for i_episode in range(1, num_episodes+1):
    state = gridworld.reset()
    episode_score = 0
    episode_actions = []
    while True:
        action = agent.act(state, epsilon=epsilon)
        new_state, reward, done, debug = gridworld.step(action)

        episode_score += reward

        new_state_action = agent.act(new_state)
        agent.learn(state, action, reward, new_state, new_state_action)

        state = new_state
        episode_actions.append(action)
        if done:
            break

    sarsa_episode_scores.append(episode_score)
    # Decay epsilon
    epsilon = max(epsilon * eps_decay_factor, eps_min)

    # For best episode data
    if episode_score > best_score:
        best_score = episode_score
        best_path_actions = episode_actions
        best_score_episodes_taken = i_episode

    print(
        f'\rEpisode: {i_episode}/{num_episodes}, score: {episode_score}, Average(last 20): {sum(sarsa_episode_scores[:-20])/len(sarsa_episode_scores)}, epsilon: {epsilon}', end='')

print(
    f'\nAfter {num_episodes}, average score: {sum(sarsa_episode_scores)/len(sarsa_episode_scores)}, Average(last 20): {sum(sarsa_episode_scores[:-20])/len(sarsa_episode_scores)}')
print(
    f'Best score: {best_score}, Sequence of actions: {[action for action in best_path_actions]}, Reached in {best_score_episodes_taken} episodes')
print('===================================================================================================================')
results_sarsa = gridworld.rewards

# <------------Q-Learning ---------------------->
print('Stats for Q Learning')
print('===================================================================================================================')
epsilon = 1
gridworld = WarehouseEnv(config)
actions = gridworld.actions.actions_extended

agent = QAgent(actions=actions, alpha=alpha,
               gamma=alpha, random_seed=random_seed)

# Storing the path taken and score for the best episode
best_score = -math.inf
best_path_actions = list()
best_score_episodes_taken = 0

for i_episode in range(1, num_episodes+1):
    state = gridworld.reset()
    episode_score = 0
    episode_actions = []
    while True:
        action = agent.act(state, epsilon=epsilon)
        new_state, reward, done, debug = gridworld.step(action)

        episode_score += reward

        agent.learn(state, action, reward, new_state)

        state = new_state
        episode_actions.append(action)
        if done:
            break

    q_episode_scores.append(episode_score)
    # Decay epsilon
    epsilon = max(epsilon * eps_decay_factor, eps_min)

    # For best episode data
    if episode_score > best_score:
        best_score = episode_score
        best_path_actions = episode_actions
        best_score_episodes_taken = i_episode

    print(
        f'\rEpisode: {i_episode}/{num_episodes}, score: {episode_score}, Average(last 20): {sum(q_episode_scores[:-20])/len(q_episode_scores)}, epsilon: {epsilon}', end='')

print(
    f'\nAfter {num_episodes}, average score: {sum(q_episode_scores)/len(q_episode_scores)}, Average(last 20): {sum(q_episode_scores[:-20])/len(q_episode_scores)}')
print(
    f'Best score: {best_score}, Sequence of actions: {[action for action in best_path_actions]}, Reached in {best_score_episodes_taken} episodes')
print('===================================================================================================================')
results_q_learning = gridworld.rewards

plt.plot(range(len(q_episode_scores)), q_episode_scores, label='Q Learning')
plt.plot(range(len(sarsa_episode_scores)), sarsa_episode_scores, label='Sarsa')
plt.legend(loc="lower right")
plt.xlabel('Episodes ->')
plt.ylabel('Score ->')
plt.title('Training progress')
plt.show()


rew_h_v4 = heuristic(config, count=1000, version='v4')

plt.xlabel('Epochen')
plt.ylabel('∅-Reward pro Step')
plt.title('Bestellung alle 4 Steps')
plt.plot(results_q_learning.get_smooth_all_episode_rewards_per_step(),
         label='q-learn')
plt.plot(results_sarsa.get_smooth_all_episode_rewards_per_step(),
         label='sarsa')
plt.plot(
    rew_h_v4.get_smooth_all_episode_rewards_per_step(), label='heur-v4')
plt.legend()
plt.show()
