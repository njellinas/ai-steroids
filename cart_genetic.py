import gym
import numpy
'''
inputs: [x, x', theta, theta']
output: 1 = right, 0 = left
'''


def genetics(pop, selection_count=1, mutation_rate=1.0):
    pop_size = len(pop)
    if selection_count < 1:
        selection_count = 1
    # print('popsize={}'.format(pop_size))
    selected = sorted(pop, key=lambda genome: genome[1], reverse=True)[:int(selection_count)]

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
            mutations = numpy.random.uniform(size=len(genome[0]), low=-mutation_rate, high=mutation_rate)
            # print('mutations={}'.format(mutations))
            new_child = child + mutations
            new_child = numpy.clip(new_child, -1, 1)
            new_child = (new_child, 0)
            # print('new_child={}'.format(new_child))
            new_pop.append(new_child)
            if len(new_pop) == pop_size:
                break
    
    new_pop.extend(selected)
    return new_pop

env = gym.make('CartPole-v1')
env._max_episode_steps = 500
GENERATIONS = 20
POPSIZE = 50
RENDER = False
pop = [(numpy.array([numpy.random.random(4) * 2 - 1]), 0) for i in range(POPSIZE)]
for i_generation in range(GENERATIONS):
    print('Starting generation {}'.format(i_generation))
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
    pop = genetics(pop, selection_count=5, mutation_rate=0.01)
# mypop = [
#     (numpy.array([0.1,0.5]),31),
#     (numpy.array([0.5,0.86]),315),
#     (numpy.array([0.1,0.5]),549),
#     (numpy.array([0.53,0.84]),138),
#     (numpy.array([0.12,0.1]),20),
#     (numpy.array([0.35,-0.1]),358),
#     (numpy.array([0.5,-0.68]),315),
#     (numpy.array([0.88,-0.8]),8),
#     (numpy.array([0.8,-0.98]),84),
#     (numpy.array([0.16,-0.111]),41)
# ]