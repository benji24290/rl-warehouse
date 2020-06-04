try:
    from envs.warehouse_env_dir.logger import log
except ModuleNotFoundError:
    from logger import log

import matplotlib.pyplot as plt
import numpy as np


class Rewards():
    def __init__(self, steps):
        self.steps = steps
        self.all_episode_rewards = []
        self.all_episode_rewards_per_step = []
        self.all_rewards_loop_storage = []
        self.all_rewards_loop_request_updates = []
        self.all_rewards_loop_arrival = []
        self.all_rewards_action_deliver = []
        self.all_rewards_action_store = []
        self.all_rewards_action_order = []
        self.step = 0
        self.step_reward = 0
        self.total_episode_reward = 0
        self.rewards_loop_storage = []
        self.rewards_loop_request_updates = []
        self.rewards_loop_arrival = []
        self.rewards_action_deliver = []
        self.rewards_action_store = []
        self.rewards_action_order = []

    def reset_episode(self):
        if self.step > 0:
            self.all_episode_rewards.append(self.total_episode_reward)
            self.all_episode_rewards_per_step.append(
                self.total_episode_reward/self.steps)
            self.all_rewards_loop_storage.append(self.rewards_loop_storage)
            self.all_rewards_loop_request_updates.append(
                self.rewards_loop_request_updates)
            self.all_rewards_loop_arrival.append(self.rewards_loop_arrival)
            self.all_rewards_action_deliver.append(self.rewards_action_deliver)
            self.all_rewards_action_store.append(self.rewards_action_store)
            self.all_rewards_action_order.append(self.rewards_action_order)
            self.step = 0
            self.step_reward = 0
            self.total_episode_reward = 0
            self.rewards_loop_storage = []
            self.rewards_loop_request_updates = []
            self.rewards_loop_arrival = []
            self.rewards_action_deliver = []
            self.rewards_action_store = []
            self.rewards_action_order = []

    def add_reward_loop_storage(self, reward):
        self.rewards_loop_storage.append(reward)

    def add_reward_loop_request_updates(self, reward):
        self.rewards_loop_request_updates.append(reward)

    def add_reward_loop_arrival(self, reward):
        self.rewards_loop_arrival.append(reward)

    def add_reward_action_deliver(self, reward):
        self.rewards_action_deliver.append(reward)

    def add_reward_action_store(self, reward):
        self.rewards_action_store.append(reward)

    def add_reward_action_order(self, reward):
        self.rewards_action_order.append(reward)

    def get_pos_neg_rewards(self, array):
        pos = 0
        neg = 0
        for v in array:
            if v > 0:
                pos += 1
            elif (v < 0):
                neg -= 1
        return [pos, neg]

    def calculate_step_reward(self):
        self.step_reward = 0
        self.step_reward += self.get_step_reward_of_array(
            self.rewards_loop_storage)
        self.step_reward += self.get_step_reward_of_array(
            self.rewards_loop_request_updates)
        self.step_reward += self.get_step_reward_of_array(
            self.rewards_loop_arrival)
        self.step_reward += self.get_step_reward_of_array(
            self.rewards_action_deliver)
        self.step_reward += self.get_step_reward_of_array(
            self.rewards_action_store)
        self.step_reward += self.get_step_reward_of_array(
            self.rewards_action_order)
        # self.episode_rewards.append(self.step_reward)
        self.total_episode_reward += self.step_reward
        self.step += 1
        return self.step_reward

    def get_total_episode_reward(self):
        return self.total_episode_reward

    def print_final_reward_infos(self):
        self.reset_episode()
        print('________________')
        print('Mean:', np.mean(self.all_episode_rewards))
        print('First episode rew:',
              self.all_episode_rewards[0])
        print('Last episode rew:',
              self.all_episode_rewards[len(self.all_episode_rewards)-1])

    def get_step_reward_of_array(self, array):
        try:
            step_reward = array[self.step]
            return step_reward
        except IndexError:
            array.append(0)
            log.debug('get_step_reward_of_array:', 'no index')
            return 0

    def plot_total_episode_rewards(self):
        '''Plots the total reward for each episode'''
        self.reset_episode()
        plt.plot(self.all_episode_rewards)
        plt.show()

    def plot_episode_rewards(self, episode):
        '''Plots the all rewards for a given episode'''
        self.reset_episode()
        try:
            plt.plot(
                self.all_rewards_loop_storage[episode], label='l_storage')
            plt.plot(
                self.all_rewards_loop_request_updates[episode], label='l_request')
            plt.plot(self.all_rewards_loop_arrival[episode], label='l_arrival')
            plt.plot(
                self.all_rewards_action_deliver[episode], label='a_deliver')
            plt.plot(self.all_rewards_action_store[episode], label='a_store')
            plt.plot(self.all_rewards_action_order[episode], label='a_order')
        except:
            log.error('plot_episode_rewards:', 'no valid episode', episode)
        plt.legend()
        plt.show()

    def plot_loot_storage(self, episode=None):
        '''Plots the loot_storage reward foreach step in each episode.
        if episode is defined only given episode will be plotted'''
        self.reset_episode()
        try:
            plt.plot(self.all_rewards_loop_storage[episode])
        except:
            for episode in self.all_rewards_loop_storage:
                plt.plot(episode)
        plt.show()

    def plot_loot_request_updates(self, episode=None):
        '''Plots the loop_request_updates reward foreach step in each episode.
        if episode is defined only given episode will be plotted'''
        self.reset_episode()
        try:
            plt.plot(self.all_rewards_loop_request_updates[episode])
        except:
            for episode in self.all_rewards_loop_request_updates:
                plt.plot(episode)
        plt.show()

    def plot_loot_arrival(self, episode=None):
        '''Plots the loot_arrival reward foreach step in each episode.
        if episode is defined only given episode will be plotted'''
        self.reset_episode()
        try:
            plt.plot(self.all_rewards_loop_arrival[episode])
        except:
            for episode in self.all_rewards_loop_arrival:
                plt.plot(episode)
        plt.show()

    def plot_action_deliver(self, episode=None):
        '''Plots the action_deliver reward foreach step in each episode.
        if episode is defined only given episode will be plotted'''
        self.reset_episode()
        try:
            plt.plot(self.all_rewards_action_deliver[episode])
        except:
            for episode in self.all_rewards_action_deliver:
                plt.plot(episode)
        plt.show()

    def plot_action_store(self, episode=None):
        '''Plots the action_store reward foreach step in each episode.
        if episode is defined only given episode will be plotted'''
        self.reset_episode()
        try:
            plt.plot(self.all_rewards_action_store[episode])
        except:
            for episode in self.all_rewards_action_store:
                plt.plot(episode)
        plt.show()

    def plot_action_order(self, episode=None):
        '''Plots the action_order reward foreach step in each episode.
        if episode is defined only given episode will be plotted'''
        self.reset_episode()
        try:
            plt.plot(self.all_rewards_action_order[episode])
        except:
            for episode in self.all_rewards_action_order:
                plt.plot(episode)
        plt.show()

    def get_loop_storage_pnr(self):
        pos = []
        neg = []
        for arr in self.all_rewards_loop_storage:
            pos.append(self.get_pos_neg_rewards(arr)[0])
            neg.append(self.get_pos_neg_rewards(arr)[1])
        return [pos, neg]

    def get_loot_request_updates_pnr(self):
        pos = []
        neg = []
        for arr in self.all_rewards_loop_request_updates:
            pos.append(self.get_pos_neg_rewards(arr)[0])
            neg.append(self.get_pos_neg_rewards(arr)[1])
        return [pos, neg]

    def get_loop_arrival_pnr(self):
        pos = []
        neg = []
        for arr in self.all_rewards_loop_arrival:
            pos.append(self.get_pos_neg_rewards(arr)[0])
            neg.append(self.get_pos_neg_rewards(arr)[1])
        return [pos, neg]

    def get_action_deliver_pnr(self):
        pos = []
        neg = []
        for arr in self.all_rewards_action_deliver:
            pos.append(self.get_pos_neg_rewards(arr)[0])
            neg.append(self.get_pos_neg_rewards(arr)[1])
        return [pos, neg]

    def get_action_store_pnr(self):
        pos = []
        neg = []
        for arr in self.all_rewards_action_store:
            pos.append(self.get_pos_neg_rewards(arr)[0])
            neg.append(self.get_pos_neg_rewards(arr)[1])
        return [pos, neg]

    def get_action_order_pnr(self):
        pos = []
        neg = []
        for arr in self.all_rewards_action_order:
            pos.append(self.get_pos_neg_rewards(arr)[0])
            neg.append(self.get_pos_neg_rewards(arr)[1])
        return [pos, neg]

    def plot_pos_neg_rewards(self, name='Pos-Neg Rewards'):
        plt.title = name
        plt.ylabel('Rewards')
        plt.xlabel('Episodes')
        plt.plot(
            self.get_loop_storage_pnr()[0], label='l_storage_p', color='blue')
        plt.plot(
            self.get_loop_storage_pnr()[1], label='l_storage_n', color='blue', linestyle='dashed')
        plt.plot(
            self.get_loot_request_updates_pnr()[0], label='l_request_p', color='orange')
        plt.plot(
            self.get_loot_request_updates_pnr()[1], label='l_request_n', color='orange', linestyle='dashed')
        plt.plot(
            self.get_loop_arrival_pnr()[0], label='l_arrival_p', color='red')
        plt.plot(
            self.get_loop_arrival_pnr()[1], label='l_arrival_n', color='red', linestyle='dashed')
        plt.plot(
            self.get_action_deliver_pnr()[0], label='a_deliver_p', color='green')
        plt.plot(
            self.get_action_deliver_pnr()[1], label='a_deliver_n', color='green', linestyle='dashed')
        plt.plot(
            self.get_action_store_pnr()[0], label='a_store_p', color='purple')
        plt.plot(
            self.get_action_store_pnr()[1], label='a_store_n', color='purple', linestyle='dashed')
        plt.plot(
            self.get_action_order_pnr()[0], label='a_order_p', color='pink')
        plt.plot(
            self.get_action_order_pnr()[1], label='a_order_n', color='pink', linestyle='dashed')
        plt.legend()
        plt.show()

    def smoothList(self, list, degree=10):
        smoothed = [0]*(len(list)-degree+1)
        for i in range(len(smoothed)):
            smoothed[i] = sum(list[i:i+degree])/float(degree)
        return smoothed
