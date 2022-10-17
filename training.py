#-----------------------------------imports------------------------------------
import gym
from itertools import product
import numpy as np
from tqdm import tqdm
import pickle

#---------------------------------environment----------------------------------
env = gym.make('Blackjack-v1', natural=True, render_mode=None)
obs_set = product(*[range(32), range(11), [True, False]])

#-----------------------------------episode------------------------------------
def episode(env, q_tbl):

    state, info = env.reset()
    terminated = False

    while not terminated:
        # exploration
        # if random number < epsilon, take a random action
        if np.random.rand() < epsilon:
          action = env.action_space.sample()
          
        # exploitation
        # else, take the action with the highest value in the current state
        else:
          action = np.argmax(q_tbl[state])
             
        # implement this action and move the agent in the desired direction
        new_state, reward, terminated, trunc, info = env.step(action)
        
        # update Q(s,a)
        q_tbl[state][action] = (1 - alpha)*q_tbl[state][action] + alpha * (
                                reward + gamma * np.max(q_tbl[new_state]) -
                                q_tbl[state][action])
        # update our current state
        state = new_state

    return q_tbl
#-------------------------------hyperparameters--------------------------------
n_episodes = 10**5
alpha = 0.01 # learning rate
gamma = 0.01 # discount factor
epsilon = 1 # amount of randomness in the action selection

#-----------------------------------training-----------------------------------
q_tbl = {obs: [0, 0] for obs in obs_set}

for ep in tqdm(range(n_episodes)):

    q_tbl = episode(env, q_tbl)

    # update parameters
    if ep > 0.7 * n_episodes:
        epsilon -= (0.1 * epsilon) / (0.3 * n_episodes) 
    elif ep > 0.3 * n_episodes:
        epsilon -= (0.8 * epsilon) / (0.4 * n_episodes)
    elif ep >= 0:
        epsilon -= (0.1 * epsilon) / (0.3 * n_episodes) 
    else:
        epsilon = 0.0
        alpha = 0.0

env.close()
#------------------------------------policy------------------------------------
policy = {obs: np.argmax(q_tbl[obs]) for obs in q_tbl}

with open('policy.pickle', 'wb') as pickle_file:
    pickle.dump(policy, pickle_file)
print('polcy saved')
print('hit percent:', sum(list(policy.values())))
