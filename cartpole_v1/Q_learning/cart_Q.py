import gym
import numpy as np
from QLearn import QLearn
import pickle
'''
inputs: [x, x', theta, theta']
output: 1 = right, 0 = left
'''


env = gym.make('CartPole-v1')
MAX_T = 250
NUM_EPISODES = 100000
alpha = 0.5
gamma = 0.9
env._max_episode_steps = MAX_T
LOAD_PRETRAINED = True

TRAIN = False

RENDER = False
bucketed_states = list()
num_buckets = [2, 2, 14, 6]
mins = [-10, -10, -0.5, -0.5]
maxes = [10, 10, 0.5, 0.5]
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
        observation = env.reset()

        for t in range(MAX_T):
            if RENDER:
                env.render()
            action = np.random.randint(env.action_space.n)  # *
            observation, reward, done, info = env.step(action)

            state_as_buckets = tuple([QLearn.matchBucket(observation[i], bucketed_states[i]) for i in range(observation_space_dimensionality)])
            stateaction = state_as_buckets + (action,)
            Q[stateaction] = (1 - alpha) * Q[stateaction] + alpha * (reward + gamma * np.amax(Q[state_as_buckets]))
            if done:
                if RENDER:
                    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
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
