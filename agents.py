import random
import os
import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix

class AgentsSwarm(object):
    def __init__(self, n, discount, alpha, epsilon, altruistic=False,
                 with_leader=False):
        if with_leader:
            self.n = n - 1
        else:
            self.n = n
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.space_dim = os.environ.get("DIMENSION_OF_SPACE", 2)
        self.bins_num = os.environ.get("SINGLE_AXIS_BINS", 11)
        self.state_dim = 3
        self.altruistic = altruistic
        if altruistic:
            self._altruistic_qvalues = lil_matrix(
                (int(2 ** (self.state_dim * self.n)),
                 int(self.bins_num ** (self.space_dim * self.n))))
        else:
            self._qvalues = defaultdict(lambda: defaultdict(
                lambda: defaultdict(lambda: 0)))

    def get_action(self, state, train=True):
        """
        Compute the action to take in the current state, including exploration.  
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self._get_best_action).
        """
        possible_actions = np.arange(self.bins_num ** self.space_dim)
        if train:
            epsilon = self.epsilon
        else:
            epsilon = -1
        chosen_actions = []
        for agent_i in range(self.n):
            if np.random.uniform() <= epsilon:
                chosen_action = np.random.choice(possible_actions)
            else:
                chosen_action = self._get_best_action(agent_i, state[agent_i])
            chosen_actions.append(chosen_action)
        return chosen_actions

    def altruistic_get_action(self, state, train=True):
        if train:
            epsilon = self.epsilon
        else:
            epsilon = -1
        possible_actions = np.arange(self.bins_num ** self.space_dim)
        if np.random.uniform() <= epsilon:
            chosen_actions = [
                np.random.choice(possible_actions) for _ in range(self.n)]
        else:
            chosen_actions = self._get_best_altruistic_action(state)
        return chosen_actions

    def get_qvalue(self, agent_num, state, action):
        return self._qvalues[agent_num][state][action]

    def set_qvalue(self, agent_num, state, action, value):
        self._qvalues[agent_num][state][action] = value

    def get_value(self, agent_num, state):
        """
        Compute the agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        """
        possible_actions = np.arange(self.bins_num ** self.space_dim)

        qvalues = [self.get_qvalue(agent_num, state, action)
                   for action in possible_actions]
        value = max(qvalues)
        return value

    def get_altruistic_value(self, state):
        """
        Compute the agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions
        in altruistic case.
        """
        value = self._altruistic_qvalues[state, :].tocoo().max()
        return value

    def update(self, state, action, reward, next_state):
        """
        Do Q-Value update here (for every agent):
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha
        for agent_i in range(self.n):
            state_i, action_i = state[agent_i], action[agent_i]
            next_state_i = next_state[agent_i]
            reward_i = reward[agent_i]
            Q_sa_i = self.get_qvalue(agent_i, state_i, action_i)
            V_next_i = self.get_value(agent_i, next_state_i)
            # V_i = self.get_value(agent_i, state_i)
            Q_sa_i = ((1 - learning_rate) * Q_sa_i +
                      learning_rate * (reward_i + gamma * V_next_i))

            self.set_qvalue(agent_i, state_i, action_i, Q_sa_i)

    def altruistic_update(self, state, action, reward, next_state):
        gamma = self.discount
        learning_rate = self.alpha

        altruistic_state = self._convert_state_to_altruistic_case(state)
        next_altruistic_state = self._convert_state_to_altruistic_case(
            next_state)
        altruistic_action = self._convert_action_to_altruistic_case(action)
        altruistic_reward = self._compute_altruistic_reward(reward)
        Q_sa = self._altruistic_qvalues[altruistic_state, altruistic_action]
        V_next = self.get_altruistic_value(next_altruistic_state)
        Q_sa = ((1 - learning_rate) * Q_sa +
                learning_rate * (altruistic_reward + gamma * V_next))
        self._altruistic_qvalues[altruistic_state, altruistic_action] = np.float(Q_sa)

    def _convert_state_to_altruistic_case(self, state):
        base = 2 ** self.state_dim
        altruistic_state = 0
        for i, state_i in enumerate(state):
            altruistic_state += (base ** i) * state_i
        return int(altruistic_state)

    def _convert_action_to_altruistic_case(self, action):
        base = self.bins_num ** self.space_dim
        altruistic_action = 0
        for i, action_i in enumerate(action):
            altruistic_action += (base ** i) * action_i
        return int(altruistic_action)

    def _convert_action_from_altruistic(self, altruistic_action):
        base = self.bins_num ** self.space_dim
        actions = []
        for _ in range(self.n - 1):
            action = altruistic_action % base
            actions.append(action)
            altruistic_action = altruistic_action // base
        actions.append(altruistic_action)
        return actions

    def _compute_altruistic_reward(self, reward):
        total_reward = 0
        for reward_i in reward:
            total_reward += reward_i
        return total_reward

    def _get_best_action(self, agent_num, state):
        """
        Compute the best action to take in a state (using current q-values). 
        """
        possible_actions = list(np.arange(self.bins_num ** self.space_dim))
        actions_qvalues = {action: self.get_qvalue(agent_num, state, action)
                           for action in possible_actions}
        best_action = max(actions_qvalues.items(), key=lambda a_q: a_q[1])[0]
        return best_action

    def _get_best_altruistic_action(self, state):
        altruistic_state = self._convert_state_to_altruistic_case(state)

        best_altruistic_action = (
            self._altruistic_qvalues[state, :].tocoo().argmax(axis=1)[0, 0])
        best_actions = self._convert_action_from_altruistic(
            best_altruistic_action)
        return best_actions
