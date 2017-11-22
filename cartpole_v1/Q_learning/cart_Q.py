import gym
import numpy
'''
inputs: [x, x', theta, theta']
output: 1 = right, 0 = left
'''


env = gym.make('CartPole-v1')
env._max_episode_steps = 500

RENDER = False


observation = env.reset()

for t in range(env._max_episode_steps):
    if RENDER:
        env.render()
    x = numpy.array(observation)
    logp = numpy.dot(W, x)
    p = 1.0 / (1.0 + numpy.exp(-logp)) # sigmoid function (gives probability of going up)
    if p > 0.5:
        action = 1
    else:
        action = 0
    observation, reward, done, info = env.step(action)
    if done:
        if RENDER:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
        break
