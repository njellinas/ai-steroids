import gym
import numpy as np
from QLearn import QLearn
import pickle
import math
'''
inputs: [x, x', theta, theta']
output: 1 = right, 0 = left
'''

env = gym.make('CartPole-v1')
MAX_T = 250
NUM_EPISODES = 1000
MIN_ALPHA = 0.1
# alpha = 0.5
gamma = 0.9
MIN_DELTA = 0.01
def delta(t):
    return max(MIN_DELTA, min(1, 1.0 - math.log10((t+1)/25)))

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
env._max_episode_steps = MAX_T
LOAD_PRETRAINED = True
LOG = True #* False
def log_message(s):
    global LOG
    if LOG:
        print(s)

TRAIN = True

RENDER = False
bucketed_states = list()
num_buckets = [2, 2, 14, 6]
mins = [-10, -10, -0.5, -1.0]
maxes = [10, 10, 0.5, 1.0]
observation_space_dimensionality = env.observation_space.shape[0]
for i_dimension in range(observation_space_dimensionality):
    bucketed_states.append(QLearn.quantized(mins[i_dimension], maxes[i_dimension], num_buckets[i_dimension]))

# Q = np.zeros((div, div, div, div, env.action_space.n))
Q = np.zeros(tuple(num_buckets) + (env.action_space.n,))
if LOAD_PRETRAINED:
    try:
        with open('Q.pkl', 'rb') as f:
            Q = pickle.load(f)
    except FileNotFoundError:
        pass

if TRAIN:
    for i_episode in range(NUM_EPISODES):
        if i_episode % 100 == 0:
            print(i_episode)
        observation = env.reset()
        state_as_buckets = tuple([QLearn.matchBucket(observation[i], bucketed_states[i]) for i in range(observation_space_dimensionality)])
        for t in range(MAX_T):
            if RENDER:
                env.render()
            action = select_action(state_as_buckets, delta(t))
            log_message('action={}'.format(action))
            observation, reward, done, info = env.step(action)
            log_message('state={}'.format(observation))
            state_as_buckets = tuple([QLearn.matchBucket(observation[i], bucketed_states[i]) for i in range(observation_space_dimensionality)])
            log_message('state buckets={}'.format(state_as_buckets))
            log_message('Q[{}]={}'.format(state_as_buckets, Q[state_as_buckets]))
            stateaction = state_as_buckets + (action,)
            temp = (1 - alpha(t)) * Q[stateaction] + alpha(t) * (reward + gamma * np.amax(Q[state_as_buckets]))
            Q[stateaction] = temp
            log_message('Q[{}]={}'.format(stateaction, Q[stateaction]))
            if done:
                log_message("Episode {} finished after {} timesteps".format(i_episode, t+1))
                break

    with open('Q.pkl', 'wb') as f:
        pickle.dump(Q, f)

else:
    observation = env.reset()
    for t in range(MAX_T):
        env.render()
        state_as_buckets = tuple([QLearn.matchBucket(observation[i], bucketed_states[i]) for i in range(observation_space_dimensionality)])
        pair = Q[state_as_buckets]
        action = np.argmax(Q[state_as_buckets])
        
        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
