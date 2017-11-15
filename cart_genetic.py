import gym
import numpy
'''
inputs: [x, x', theta, theta']
output: 1 = right, 0 = left
'''


def genetics(pop, selection_count=1, mutation_rate=1.0):
    def mutate(vals, mutation_rate):
        mutations = numpy.random.uniform(size=len(vals), low=-mutation_rate, high=mutation_rate)
        new_vals = vals + mutations
        new_vals = numpy.clip(new_vals, -1, 1)
        return new_vals

    pop_size = len(pop)
    if selection_count < 1:
        selection_count = 1
    selected = sorted(pop, key=lambda genome: genome[1], reverse=True)[:int(selection_count)]
    new_pop = list()
    new_pop.extend(selected)
    for genome in selected:
        for i_child in range(int((pop_size - selection_count) / selection_count)):
            child = (mutate(genome[0], mutation_rate), 0)
            new_pop.append(child)
    for genome in selected[:(pop_size - selection_count) % selection_count]:
        child = (mutate(genome[0], mutation_rate), 0)
        new_pop.append(child)
    return new_pop

env = gym.make('CartPole-v1')
env._max_episode_steps = 500
GENERATIONS = 20
POPSIZE = 100
RENDER = False
pop = [(numpy.array([numpy.random.random(4) * 2 - 1]), 0) for i in range(POPSIZE)]
for i_generation in range(GENERATIONS):
    # print('Starting generation {}'.format(i_generation))
    for i_episode in range(POPSIZE):
        observation = env.reset()
        tup = pop[i_episode]
        W = tup[0]
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
                pop[i_episode] = (W, t)
                break
    mean_fitness = numpy.mean([genome[1] for genome in pop])
    print('Generation {} mean fitness: {}'.format(i_generation, mean_fitness))
    pop = genetics(pop, selection_count=4, mutation_rate=0.01)
