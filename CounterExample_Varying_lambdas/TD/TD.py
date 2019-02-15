#! /usr/bin/env python
import sys
sys.path.append("..")
from Environment.TwoStates import TwoStates
import numpy as np
import os
import argparse
from time import sleep
import copy

# lambda of state_0 is 0 and 1 for state_1
def lam(state):
    return state

# feature for these two states
def feature(state):
    if state == 0:
        return [3, 1]
    elif state == 1:
        return [1 ,1]
    else:
        assert False, 'wrong state input'


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename', default='Data/TD_results', type=str,
                    help='Data/lambda/alpha/filename')
parser.add_argument('--alpha_2RaisedTo', default='-8', type=float, # NOTICE: for convenince, the input for step size is the exponent of 2.
                    help='step size = 2 ** input!!!')
parser.add_argument('--gamma', default='0.95', type=float,
                    help='discount rate')
parser.add_argument('--num_trainingsteps', default='1200', type=int,
                    help='training steps')


args = parser.parse_args()
filename = args.filename
alpha = pow(2, args.alpha_2RaisedTo)  # initialize algorithm setting
gamma = args.gamma
max_steps = args.num_trainingsteps


def main():
    dimension = 2
    weight_vector = np.array([10000.0, 10000.0])
    env = TwoStates()

    state = env.reset()
    eligibility_trace = np.zeros(dimension)
    step = 0 # A counter of training process. Terminate when it exceeds max_steps

    weight_vectors = [copy.deepcopy(weight_vector)]
    estimated_values = [[np.dot(weight_vector, feature(0)), np.dot(weight_vector, feature(1))]]

    while step < max_steps:

        print(str(step) + 'th step: weight vector = ' + str(weight_vector))
        print('estimated value for two states = (' + str(np.dot(weight_vector, feature(0))) + ', ' + str(np.dot(weight_vector, feature(1))) + ')')
        # sleep(0.01)

        next_state, reward, _ = env.step() # No longer use done as our training counter since this problem is continuing

        # Update eligibility trace
        eligibility_trace = gamma * lam(state) * eligibility_trace + feature(state)
        # Compute TD error
        TD_error = reward + gamma * np.dot(weight_vector, feature(next_state)) - np.dot(weight_vector, feature(state))
        # Update weight vector
        weight_vector += alpha * TD_error * eligibility_trace

        state = next_state

        weight_vectors.append(copy.deepcopy(weight_vector))
        estimated_values.append([np.dot(weight_vector, feature(0)), np.dot(weight_vector, feature(1))])
        step += 1

    np.savetxt(filename + '_estimated_values', estimated_values)
    np.savetxt(filename + '_weights', weight_vectors)

if __name__ == '__main__':
    main()
