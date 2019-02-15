from Environment.MountainCar import MountainCar
from Environment.simple_policy import simple_policy
import numpy as np
import random

# Compute the root of (approximated) Mean-Squared Error
def MSE(samples, feature_samples, current_weight_vector):
    assert samples.shape[0] == feature_samples.shape[0]

    SE = 0
    # For each state, compute the value error
    for sample, feature_sample in zip(samples, feature_samples):
        # Get the true and estimated value of states in samples
        true_value = sample[2]
        estimated_value = np.dot(current_weight_vector, feature_sample)
        SE += (true_value - estimated_value) ** 2
    return np.sqrt(SE/samples.shape[0])

# Write results in MSEs to filename
def writeF(filename, MSEs):
    with open(filename, 'w') as file:
        for idx in range(len(MSEs)):
            file.write(str(MSEs[idx]) + '\n' )