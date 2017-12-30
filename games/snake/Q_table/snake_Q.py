import gym
import numpy as np
import math
import pickle

# from QLearn import QLearn


env = gym.make('Snake-v0')

MAX_T = 40000
env._max_episode_steps = MAX_T

NUM_EPISODES = 100000

gamma = 0.1

MIN_DELTA = 0.01
def delta(t):
    return max(MIN_DELTA, min(1, 1.0 - math.log10((t+1)/25)))

MIN_ALPHA = 0.1
def alpha(t):
    return max(MIN_ALPHA, min(0.5, 1.0 - math.log10((t+1)/25)))

def select_action(state, delta):
    # Select a random action
    if np.random.random() < delta:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        Q_state = np.array([Q_of(state, action) for action in range(env.action_space.n)])
        action = np.argmax(Q_state)
    return action


def log_message(s):
    global LOG
    if LOG:
        print(s)

LOAD_PRETRAINED = True
TRAIN = True #* False #**************************************************************************
LOG = True * False
RENDER = True * False



state_space_dimensionality = 3 # env.observation_space.shape[0]
# Q = np.zeros((220,2**10,220,4),dtype=np.int8)
Q = dict()
def Q_of(state, action):
    global Q
    head, shape, food = state
    key = ','.join([str(head), str(shape), str(food), str(action)])
    if key in Q.keys():
        return Q[key]
    else:
        return 0

if LOAD_PRETRAINED:
    try:
        with open(r'games\snake\Q_table\snake_Q_table.pkl', 'rb') as f:
            Q = pickle.load(f)
    except FileNotFoundError:
        pass

if TRAIN:
    scores = 0
    for i_episode in range(NUM_EPISODES):
        if i_episode % 100 == 0:
            print('ep {}'.format(i_episode))
            if i_episode > 0:
                score = scores / 100
                print('100 ep mean score: {}'.format(score))
                scores = 0
                if i_episode % 10000 == 0:
                    with open(r'games\snake\Q_table\snake_Q_table.pkl', 'wb') as f:
                        pickle.dump(Q, f)
        state = env.reset()
        rewards = 0
        for t in range(MAX_T):
            if RENDER:
                env.render()
            action = select_action(state, delta(t))

            log_message('state={}'.format(state))
            log_message('action={}'.format(action))

            state, reward, done, info = env.step(action)
            rewards += reward
            next_state = tuple([state[i] for i in range(state_space_dimensionality)])
            # stateaction = state + (action,)
            stateaction = ','.join([str(x) for x in state]+[str(action)])
            Q_next_state = [Q_of(next_state, action) for action in range(env.action_space.n)]
            Q[stateaction] = (1 - alpha(t)) * Q_of(state, action) + alpha(t) * (reward + gamma * np.amax(Q_next_state))
            log_message('Q[{}]={}'.format(stateaction, Q_of(state, action)))
            state = next_state
            if done:
                log_message("Episode {} finished after {} timesteps, total reward {}".format(i_episode, t + 1, rewards))
                scores += rewards
                break

    with open(r'games\snake\Q_table\snake_Q_table.pkl', 'wb') as f:
        pickle.dump(Q, f)

else:
    for _ in range(NUM_EPISODES):
        state = env.reset()
        for t in range(MAX_T):
            env.render()
            Q_state = [Q_of(state, action) for action in range(env.action_space.n)]
            action = np.argmax(Q_state)
            action_Q = np.amax(Q_state)
            print(action)
            print(action_Q)
            state, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
