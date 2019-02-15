#! /usr/bin/env python
import sys
sys.path.append("..")
from pylab import zeros, sin, cos, normal, random
from Tile_Coding.Tilecoder import numTilings, num_tiles_per_tiling, tilecode, feature
from Environment.MountainCar import MountainCar
from Environment.simple_policy import simple_policy
from Utility.util import MSE, writeF
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename', default='TD_results', type=str,
                    help='Data/lambda/alpha/filename')
parser.add_argument('--lam', default='1.0', type=float,
                    help='bootstrapping degree')
parser.add_argument('--alpha_2RaisedTo', default='-8', type=float, # NOTICE: for convenince, the input for step size is the exponent of 2.
                    help='step size = 2 ** input!!!') 
parser.add_argument('--gamma', default='1.0', type=float,
                    help='discount rate')
parser.add_argument('--num_episodes', default='50000', type=int,
                    help='training episodes')                 

# Interact with the MountainCar environment using the simple_policy
# with a linear function approximator (with tile coder) to learn a 
# function approximator that evaluate the state-values of this policy

args = parser.parse_args()

alpha = pow(2, args.alpha_2RaisedTo)  # initialize algorithm setting
lam = args.lam
gamma = args.gamma
max_episodes = args.num_episodes

# Find the feature of states in samples
# Note: samples is a list of [position, velocity, true_value]s
def create_features(samples):
    assert samples.shape[1] == 3

    result = np.zeros(shape=(samples.shape[0], num_tiles_per_tiling * numTilings))
    # For each state, compute its feature
    for idx in range(samples.shape[0]):
        sample = samples[idx]
        feature_sample = feature(sample)
        result[idx] = feature_sample
    return result

def main():
    # Load the 500 sample states and their true state value
    samples = np.load('sampleOnPolicy.npy')
    feature_samples = create_features(samples)

    dimension =  num_tiles_per_tiling * numTilings

    weight_vector = np.zeros(dimension)
    MSEs = np.zeros(max_episodes)
    env = MountainCar()

    for idx in range(max_episodes):
        # Compute MSVE at the begining of each episode
        MSEs[idx] = MSE(samples, feature_samples, weight_vector)
        print('MSE of ' + str(idx) + 'th episodes: ' + str(MSEs[idx]))

        state = env.reset()
        eligibility_trace = np.zeros(dimension)
        done = False
        while not done:
            action = simple_policy(state[1])
            next_state, reward, done = env.step(action)

            # Update eligibility trace
            eligibility_trace = gamma * lam * eligibility_trace + feature(state)
            # Compute TD error
            TD_error = reward + gamma * np.dot(weight_vector, feature(next_state)) - np.dot(weight_vector, feature(state))
            # Update weight vector
            weight_vector += alpha * TD_error * eligibility_trace
            
            state = next_state

    # Write out result (MSEs) to a file in target directory
    writeF(args.filename, MSEs)
    
if __name__ == '__main__':
    main()








