#-----------------------------------imports------------------------------------
import gym
import numpy as np
import pickle
from tqdm import tqdm

#----------------------------------q-episode-----------------------------------
# plays an episode using q-table and returns reward
def play_q_ep(env, policy):

    state, info = env.reset()
    terminated = False

    while not terminated:

        # take the action with the highest value in the current state
        action = policy[state]
             
        # implement this action and move the agent in the desired direction
        new_state, reward, terminated, trunc, info = env.step(action)
        
        # update our current state
        state = new_state

    return reward

#--------------------------------random episode--------------------------------
# plays an episode using a random strategy and returns reward
def play_r_ep(env, q_tbl):

    state, info = env.reset()
    terminated = False
    while not terminated:

        # take the action with the highest value in the current state
        action = env.action_space.sample()
             
        # implement this action and move the agent in the desired direction
        new_state, reward, terminated, trunc, info = env.step(action)
        
        # update our current state
        state = new_state

    return reward

#---------------------------------environment----------------------------------
env = gym.make('Blackjack-v1', natural=True, render_mode=None)

#------------------------------------policy------------------------------------
with open('policy.pickle', 'rb') as pickle_file:
    policy = pickle.load(pickle_file)

#----------------------------------evaluation----------------------------------
n_episodes = 10**4
bankroll_init = 100

win_dict = {'policy': [play_q_ep(env, policy)
                  for ep in tqdm(range(n_episodes), desc = 'policy')],
            'random': [play_r_ep(env, policy) 
                       for ep in tqdm(range(n_episodes), desc = 'random')]}

policy_win_rate = sum(np.array(win_dict['policy']) > 0)/n_episodes
random_win_rate = sum(np.array(win_dict['random']) > 0)/n_episodes

print(f'policy win rate: {policy_win_rate:.0%}', '|',
      f'random win rate: {random_win_rate:.0%}')
