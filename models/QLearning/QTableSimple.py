import logging

import numpy as np
import math


class QTableSimple(object):
    def __init__(self, state_dim, action_dim):
        # Initialize basic variables
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize Q variables default values
        self.min_er = 0.01  # Exploration rate
        self.min_lr = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor

        # Set Q variables to initial values
        self.er = self.get_exploration_rate(0)
        self.lr = self.get_learning_rate(0)
        self.qtable = np.zeros((self.state_dim, self.action_dim))

        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def update(self, episode, old_state, action, new_state, reward):
        # Find maximum Q reward
        best_q = np.amax(self.qtable[new_state])
        # Update Q table
        self.qtable[old_state, action] += self.lr*(
            reward + self.gamma*best_q - self.qtable[old_state, action]
        )
        # Get current exploration and learning rates
        self.er = self.get_exploration_rate(episode)
        self.lr = self.get_learning_rate(episode)

    def set_q_parameters(self, min_er=0.01, min_lr=0.1, gamma=0.99):
        self.min_er = min_er
        self.min_lr = min_lr
        self.gamma = gamma

    def get_exploration_rate(self, episode):
        return max(self.min_er, min(1.0, 1.0 - math.log10((episode+1)/25)))

    def get_learning_rate(self, episode):
        return max(self.min_lr, min(0.5, 1.0 - math.log10((episode+1) / 25)))

    def reset(self):
        self.er = self.min_er
        self.lr = self.min_lr
        self.qtable = np.zeros((self.state_dim, self.action_dim))
