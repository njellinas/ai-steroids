import numpy as np
import logging


class Population(object):
    def __init__(self, size=50, generations=20, chromosome_shape=4):
        self.size = size
        self.generations = generations
        self.chromosome_shape = chromosome_shape

        # Initialize the logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize the population with random values
        self.population = self.initialize_population()

    def initialize_population(self):
        self.logger.info('Initialized population of size {} with chromosome shape {}'.
                         format(self.size, self.chromosome_shape))
        return [(np.random.uniform(low=-1, high=1, size=self.chromosome_shape), 0) for _ in range(self.size)]

    def genetics(self, selection_count=1, mutation_rate=1.0):
        pop_size = self.size
        if selection_count < 1:
            selection_count = 1
        # print('popsize={}'.format(pop_size))
        selected = sorted(self.population, key=lambda genome: genome[1], reverse=True)[:int(selection_count)]

        # print('selected={}'.format(selected))
        num_weigths = len(selected[0][0])
        # print('num_weigths={}'.format(num_weigths))
        new_pop = list()

        for genome in selected:
            # print('genome: {}'.format(genome))
            if len(new_pop) == pop_size:
                # print('got in here?')
                break
            for i_child in range(int(pop_size / selection_count) - 1):
                child = genome[0]
                # print('genome={}'.format(genome))
                # print('genome[0]={}'.format(genome[0]))
                # print('child={}'.format(child))
                mutations = np.random.uniform(size=len(genome[0]), low=-mutation_rate, high=mutation_rate)
                # print('mutations={}'.format(mutations))
                new_child = child + mutations
                new_child = np.clip(new_child, -1, 1)
                new_child = (new_child, 0)
                # print('new_child={}'.format(new_child))
                new_pop.append(new_child)
                if len(new_pop) == pop_size:
                    break
        new_pop.extend(selected)
        self.population = new_pop


