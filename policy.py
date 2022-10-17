#-----------------------------------imports------------------------------------
import gym
from itertools import product
from tqdm import tqdm
import pickle

#---------------------------------environment----------------------------------
env = gym.make('Blackjack-v1', natural=True, render_mode=None)
obs_set = product(*[range(32), range(11), [True, False]])

obs_set = [env.reset()[0] for i in tqdm(range(10**5))]

#------------------------------------policy------------------------------------
policy = {}
for obs in obs_set:
    if obs[2] == True:
        if obs[0] <= 17:
            action = 1
        elif obs[0] == 18 and obs[1] in (9, 10, 1):
            action = 1
        else:
            action = 0
    else:
        if obs[0] <= 11:
            action = 1
        elif obs[0] == 12 and obs[1] not in (4, 5, 6):
            action = 1
        elif obs[0] >= 13 and obs[0] <= 16 and obs[1] not in range(2,7):
            action = 1
        else:
            action = 0

    policy[obs] = action

with open('policy_0.pickle', 'wb') as pickle_file:
    pickle.dump(policy, pickle_file)
print('polcy saved')
print('hit percent:', sum(list(policy.values())))