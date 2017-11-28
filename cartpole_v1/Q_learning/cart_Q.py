import gym
import numpy
from QLearn import *
'''
inputs: [x, x', theta, theta']
output: 1 = right, 0 = left
'''


env = gym.make('CartPole-v1')
env._max_episode_steps = 500

RENDER = False
bucketed_observations = list()
div = 10
observation_space_dimensionality = env.observation_space.shape[0]
for i_dimension in range(observation_space_dimensionality):
    bucketed_observations.append(QLearn.quantized(-5, 5, div))

Q = numpy.zeros((div, div, div, div, env.action_space.n))
observation = env.reset()
action = numpy.random.random_integers(0, 1)
observation_as_buckets = [QLearn.matchBucket(observation[i], bucketed_observations[i]) for i in range(observation_space_dimensionality)]
stateaction = tuple(observation_as_buckets) + (action,)
Q[stateaction] = reward + gamma * max( next_state_all_actions_future_Q )


#for t in range(env._max_episode_steps):
while not done:
    if RENDER:
        env.render()

    observation, reward, done, info = env.step(action)
    if done:
        if RENDER:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
        break
