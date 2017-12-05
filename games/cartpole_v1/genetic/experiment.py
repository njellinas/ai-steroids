import gym
import numpy as np
import logging

from models.genetic_algorithms.population import Population
from models.genetic_algorithms.algorithms import logistic

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize CartPole-v1 simulator
env = gym.make('CartPole-v1')
max_episodes = 500
env._max_episode_steps = max_episodes
RENDER = False

pop = Population(size=50,
                 generations=20,
                 chromosome_shape=4)

for i_generation in range(pop.generations):
    logger.info('Starting generation {}'.format(i_generation))
    for i_chromosome in range(pop.size):
        observation = env.reset()
        tup = pop.population[i_chromosome]
        W = tup[0]
        for t in range(max_episodes):
            if RENDER:
                env.render()
            x = np.array(observation)
            action = logistic(w=W, x=x)
            observation, reward, done, info = env.step(action)
            if done:
                if RENDER:
                    logger.info("Episode {} finished after {} timesteps".format(i_chromosome, t + 1))
                pop.population[i_chromosome] = (W, t)
                break
    mean_fitness = np.mean([genome[1] for genome in pop.population])
    logger.info('Generation {} mean fitness: {}'.format(i_generation, mean_fitness))
    pop.genetics(selection_count=5, mutation_rate=0.01)
