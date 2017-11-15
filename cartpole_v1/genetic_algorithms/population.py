import numpy as np
import logging


class Population(object):
    def __init__(self, size=50, generations=20, chromosome_shape=4):
        self.size = size
        self.generations = generations
        self.chromosome_shape = chromosome_shape

        # Initialize the population with random values
        self.population = self.initialize_population()

        # Initialize the logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize_population(self):
        self.logger.info('Initialized population of size {} with chromosome shape {}'.
                         format(self.size, self.chromosome_shape))
        return [(np.random.uniform(low=-1, high=1, size=self.chromosome_shape), 0) for _ in range(self.size)]
