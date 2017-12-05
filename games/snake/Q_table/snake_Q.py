import gym
import numpy as np
import math
import pickle

from QLearn import QLearn


env = gym.make('Snake-v0')

MAX_T = 40000
env._max_episode_steps = MAX_T

NUM_EPISODES = 10000

gamma = 0.99

MIN_DELTA = 0.01
def delta(t):
    return max(MIN_DELTA, min(1, 1.0 - math.log10((t+1)/25)))

MIN_ALPHA = 0.1
def alpha(t):
    return max(MIN_ALPHA, min(0.5, 1.0 - math.log10((t+1)/25)))

def select_action(state, delta):
    # Select a random action
    if np.random.random() < delta:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(Q[state])
    return action


LOG = True * False
def log_message(s):
    global LOG
    if LOG:
        print(s)

TRAIN = True #* False #**************************************************************************
LOAD_PRETRAINED = False

RENDER = False


Q = np.zeros(tuple(env.observation_space.n) + (env.action_space.n,))

if LOAD_PRETRAINED:
    try:
        with open('Q_table.pkl', 'rb') as f:
            Q = pickle.load(f)
    except FileNotFoundError:
        pass

if TRAIN:
    scores = 0
    for i_episode in range(NUM_EPISODES):
        if i_episode % 100 == 0:
            print('ep {}'.format(i_episode))
            if i_episode > 0:
                score = scores / 100
                print('100 ep mean score: {}'.format(score))
                scores = 0
        observation = env.reset()
        state_as_buckets = tuple([QLearn.matchBucket(observation[i], bucketed_states[i]) for i in range(observation_space_dimensionality)])
        for t in range(MAX_T):
            if RENDER:
                env.render()
            action = select_action(state_as_buckets, delta(t))

            log_message('state={}'.format(observation))
            log_message('state buckets={}'.format(state_as_buckets))
            log_message('action={}'.format(action))
            log_message('Q[{}]={}'.format(state_as_buckets, Q[state_as_buckets]))

            observation, reward, done, info = env.step(action)
            next_state_as_buckets = tuple([QLearn.matchBucket(observation[i], bucketed_states[i]) for i in range(observation_space_dimensionality)])
            stateaction = state_as_buckets + (action,)
            Q[stateaction] = (1 - alpha(t)) * Q[stateaction] + alpha(t) * (reward + gamma * np.amax(Q[next_state_as_buckets]))
            log_message('Q[{}]={}'.format(stateaction, Q[stateaction]))
            state_as_buckets = next_state_as_buckets
            if done:
                log_message("Episode {} finished after {} timesteps".format(i_episode, t + 1))
                scores += t + 1
                break

    with open('Q_table.pkl', 'wb') as f:
        pickle.dump(Q, f)

else:
    for _ in range(NUM_EPISODES):
        observation = env.reset()
        for t in range(MAX_T):
            env.render()
            state_as_buckets = tuple([QLearn.matchBucket(observation[i], bucketed_states[i]) for i in range(observation_space_dimensionality)])
            pair = Q[state_as_buckets]
            action = np.argmax(Q[state_as_buckets])
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
