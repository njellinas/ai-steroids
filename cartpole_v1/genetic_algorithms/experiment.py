import gym
import numpy as np
import logging


# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize CartPole-v1 simulator
env = gym.make('CartPole-v1')
env._max_episode_steps = 500
RENDER = False



