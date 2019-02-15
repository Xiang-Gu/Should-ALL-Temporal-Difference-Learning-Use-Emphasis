import numpy as np
import sys

''' Several utility functions for Sarsa(lambda) or ETD(lambda) in Acrobot '''

# Check for divergence (when step size is too large so that weight vector becomes inf or nan)
def is_weight_valid(weight_vector):
    if np.isinf(weight_vector).any() or np.isnan(weight_vector).any():
        print('weight vector has become invalid (overflow). Break out and finish the program \n', file=sys.stderr)
        return False
    else:
        return True
