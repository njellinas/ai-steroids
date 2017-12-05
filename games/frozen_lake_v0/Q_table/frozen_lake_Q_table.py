import logging
import gym
import numpy as np
from collections import deque
from matplotlib import pyplot as plt

from models.QLearning.QTableSimple import QTableSimple


def select_action(q_table, state, exploration_rate):
    # Select action with noise
    act = np.argmax(qtable.qtable[state, :] + np.random.randn(1, action_dim)*(1./(episode+1)))
    # Select action based on exploration rate
    # if np.random.random() < exploration_rate:
    #     act = env.action_space.sample()
    # else:
    #     act = np.argmax(q_table[state])
    return act


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = gym.make('FrozenLake-v0')

# Initialization
state_dim = env.observation_space.n
action_dim = env.action_space.n
qtable = QTableSimple(state_dim=state_dim,
                      action_dim=action_dim)
qtable.set_q_parameters(min_lr=0.9, gamma=0.95)
num_episodes = 5000
reward_list = deque(maxlen=100)
max_episode_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

for episode in range(num_episodes):
    observation = env.reset()
    done = 0
    episode_reward = 0
    old_state = observation
    for t in range(max_episode_steps):
        if done:
            break
        action = select_action(qtable.qtable, old_state, qtable.er)
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        qtable.update(episode=episode, old_state=old_state,
                      action=action, new_state=new_state,
                      reward=reward)
        old_state = new_state
    reward_list.append(episode_reward)
    if episode % 100 == 0:
        print('Last 100 episode mean reward: ' + str(sum(reward_list)/100))
# plt.plot(reward_list)
# plt.show()
