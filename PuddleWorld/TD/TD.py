#!/lusr/bin/python3
import sys
sys.path.append("..")
from Tile_Coding.Tilecoder import numTilings, num_tiles_per_tiling, tilecode, feature, num_actions
from Environment.PuddleWorld import PuddleWorld
from Util.util import epsilon_greedy, is_weight_valid
import numpy as np
import os
import argparse
from time import sleep

'''Sarsa(lambda) algorithm that optimize the policy for PuddleWorld problem'''

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename', default='TD_results', type=str, help='Data/lambda/alpha/filename')
parser.add_argument('--lam', default='0.9', type=float, help='bootstrapping degree')
# NOTICE: for convenince, the input for step size is the exponent of 2.
parser.add_argument('--alpha_2RaisedTo', default='-9', type=float, help='step size = 2 ** input!!!')
parser.add_argument('--gamma', default='1.0', type=float, help='discount rate')
parser.add_argument('--epsilon', default='0.1', type=float, help='epsilon')
parser.add_argument('--num_episodes', default='3000', type=int, help='training episodes')

# initialize algorithm setting
args = parser.parse_args()
alpha = pow(2, args.alpha_2RaisedTo)
lam = args.lam
gamma = args.gamma
epsilon = args.epsilon
max_episodes = args.num_episodes
# max_steps_per_episode = 999

def main():
    dimension = num_tiles_per_tiling * numTilings * num_actions
    weight_vector = np.zeros(dimension)
    env = PuddleWorld()
    performances = [] # Record the steps used and reward collect within each episode

    for idx in range(max_episodes):
        # Check for overflow in weight vector (happens when step size is too large)
        if not is_weight_valid(weight_vector):
            # Replace the performance measure in last episode, when the overflow occurs, to infinity as an error signal
            performances[-1] = [np.float('inf'), np.float('inf')]
            break

        state = env.reset()
        action = epsilon_greedy(state, weight_vector, epsilon)
        eligibility_trace = np.zeros(dimension)
        done = False

        print(str(idx) + 'th episode')
        steps_per_episode = 0 # steps required in this episode
        reward_per_episode = 0.0

        while not done:
            next_state, reward, done = env.step(action)

            # Update eligibility trace
            eligibility_trace = gamma * lam * eligibility_trace + feature(state, action)

            next_action = epsilon_greedy(next_state, weight_vector, epsilon)

            # Compute TD error
            if done:
                TD_error = reward - np.dot(weight_vector, feature(state, action))
            else:
                TD_error = reward + gamma * np.dot(weight_vector, feature(next_state, next_action)) - np.dot(weight_vector, feature(state, action))

            # Update weight vector
            weight_vector += alpha * TD_error * eligibility_trace

            state = next_state
            action = next_action

            steps_per_episode += 1
            reward_per_episode += reward

        print('steps used in this episode: ' + str(steps_per_episode))
        print('rewards in this episode: ' + str(reward_per_episode) + '\n')
        performances.append([steps_per_episode, reward_per_episode])

    np.savetxt(args.filename, performances)

if __name__ == '__main__':
    main()
