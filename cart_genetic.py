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
        if len(new_pop) == pop_size:
            print('got in here?')
            break
        for i_child in range(int(pop_size / selection_count) + 1):
            child = genome[0]
            # print('child={}'.format(child))
            mutations = numpy.array([(numpy.random.random() - 0.5) * mutation_rate for w in child])
            # print('mutations={}'.format(mutations))
            child += mutations
            child = numpy.clip(child, 0, 1)
            # print('new_child={}'.format(child))
            new_pop.append(child)
            if len(new_pop) == pop_size:
                break
    return new_pop


env = gym.make('CartPole-v1')
env._max_episode_steps = 500
GENERATIONS = 10
POPSIZE = 10
RENDER = True
pop = [numpy.random.random(4) for i in range(POPSIZE)]
for i_generation in range(GENERATIONS):
    print('Starting generation {}'.format(i_generation))
    for i_episode in range(POPSIZE):
        observation = env.reset()
        W = pop[i_episode]
        for t in range(env._max_episode_steps):
            if RENDER:
                env.render()
            x = observation
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
                pop[i_episode] = ((W, t))
                break
    mean_fitness = numpy.mean([genome[1] for genome in pop])
    print('Generation {} mean fitness: {}'.format(i_generation, mean_fitness))
    pop = genetics(pop, selection_count=2, mutation_rate=0.1)
