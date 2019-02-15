#! /usr/bin/env python
import sys
sys.path.append("..")
from pylab import zeros, sin, cos, normal, random
from Tile_Coding.Tilecoder import numTilings, num_tiles_per_tiling, tilecode, feature, num_actions
from Environment.MountainCar import MountainCar
import numpy as np
import os
import argparse
from time import sleep

# Query this function to get user-defined Interests to each state
# in the mountaincar environment
def interest(state):
    return 1

# Importance sampling ratio at each time step
def rho(state):
    return 1


# Use ETD(lambda) to do Control

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename', default='ETD_results', type=str,
                    help='Data/lambda/alpha/filename')
parser.add_argument('--lam', default='0.9', type=float,
                    help='bootstrapping degree')
parser.add_argument('--alpha_2RaisedTo', default='-12', type=float, # NOTICE: for convenince, the input for step size is the exponent of 2.
                    help='step size = 2 ** input!!!')
parser.add_argument('--gamma', default='1.0', type=float,
                    help='discount rate')
parser.add_argument('--epsilon', default='0.01', type=float,
                    help='epsilon')
parser.add_argument('--num_episodes', default='3000', type=int,
                    help='training episodes')

args = parser.parse_args()

alpha = pow(2, args.alpha_2RaisedTo)  # initialize algorithm setting
lam = args.lam
gamma = args.gamma
epsilon = args.epsilon
max_episodes = args.num_episodes
max_steps_per_episode = 999

# Choose an action in state epsilon_greedily
# by consulting the action-values given by a linear
# function approximator with weight_vector
def epsilon_greedy(state, weight_vector, epsilon):
    estimated_action_values = []
    # Get apprximate action-values for all legal action in state
    for action in range(num_actions):
        action = action - 1 # Remember our legal actions are -1, 0, 1
        ft = feature(state, action)
        estimated_action_value = np.dot(weight_vector, ft)
        estimated_action_values.append(estimated_action_value)

    if np.random.random() > epsilon:
        max_q = max(estimated_action_values)
        greedy_actions = [action for action, q in enumerate(estimated_action_values) if q == max_q]
        action = np.random.choice(greedy_actions) if len(greedy_actions) != 0 else 1 # if weight_vector overflows and thus max_q = nan we always return 0
        return action - 1 # Note we use -1, 0, 1 as action representations in our main code
    else:
        return np.random.choice([-1, 0, 1])

# Check for divergence (when step size is too large so that weight vector becomes inf or nan)
def is_weight_valid(weight_vector):
    if np.isinf(weight_vector).any() or np.isnan(weight_vector).any():
        print('weight vector has become invalid (overflow). Break out and finish the program \n', file=sys.stderr)
        return False
    else:
        return True

def main():
    dimension =  num_tiles_per_tiling * numTilings * num_actions
    weight_vector = np.zeros(dimension)
    env = MountainCar()
    steps = [] # A list of steps in each episode

    for idx in range(max_episodes):
        # Check for overflow in weight vector (happens when step size is too large)
        if not is_weight_valid(weight_vector):
            # Replace the performance measure in last episode, when the overflow occurs, to infinity as an error signal
            print('Invalid value encountered. Break out!')
            steps[-1] = max_steps_per_episode + 1
            break

        print(str(idx) + 'th episode')
        steps_per_episode = 0

        state = env.reset()
        action = epsilon_greedy(state, weight_vector, epsilon)

        # Initialize follow-on trace F_0
        F = interest(state)
        # Initialize Emphasis M_0
        M = lam * interest(state) + (1 - lam) * F
        # Initialize eligibility trace e_0
        eligibility_trace = rho(state) * M * feature(state, action)
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(next_state, weight_vector, epsilon)

            steps_per_episode += 1

            # Compute TD error
            if done:
                TD_error = reward - np.dot(weight_vector, feature(state, action))
            else:
                TD_error = reward + gamma * np.dot(weight_vector, feature(next_state, next_action)) - np.dot(weight_vector, feature(state, action))

            # Update weight vector
            weight_vector += alpha * TD_error * eligibility_trace

            # Update Follow-on trace
            F = rho(state) * gamma * F + interest(next_state)

            # Update Emphasis
            M = lam * interest(next_state) + (1 - lam) * F

            # Update eligibility trace
            eligibility_trace = rho(next_state) * (gamma * lam * eligibility_trace + M * feature(next_state, next_action))

            state = next_state
            action = next_action

            # Terminate current episode if exceeds max_steps_per_episode
            if steps_per_episode > max_steps_per_episode:
                break

        # Record steps used for this episode
        print('steps used in this episode: ' + str(steps_per_episode) + '\n')
        steps.append(steps_per_episode)

    np.savetxt(args.filename, steps)

if __name__ == '__main__':
    main()
