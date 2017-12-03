# Q-Learning
# http://mnemstudio.org/path-finding-q-learning-tutorial.htm

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from collections import deque

num_episodes = 1000


def run_episode(env, Q, learning_rate, discount, episode, render=False):
    observation = env.reset()
    done = False
    t_reward = 0
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    for i in range(max_steps):
        if done:
            break

        if render:
            env.render()

        curr_state = observation

        action = np.argmax(Q[curr_state, :] + np.random.randn(1, env.action_space.n) * (1. / (episode + 1)))

        observation, reward, done, info = env.step(action)

        t_reward += reward

        # Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
        Q[curr_state, action] += learning_rate * (reward + discount * np.max(Q[observation, :]) - Q[curr_state, action])

    return Q, t_reward


def train():
    env = gym.make('FrozenLake-v0')
    env = wrappers.Monitor(env, '/tmp/FrozenLake-experiment-6', force=True)
    learning_rate = 0.81
    discount = 0.96

    reward_per_ep = deque(maxlen=100)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for i in range(num_episodes):
        Q, reward = run_episode(env, Q, learning_rate, discount, i)
        reward_per_ep.append(reward)
        if i % 100 == 0:
            print(str(sum(reward_per_ep)/100))
    plt.plot(reward_per_ep)
    # plt.show()
    return Q


q = train()
# print(q)
