import numpy as np
from Tile_Coding.Tilecoder import feature, num_actions
import sys

''' Several utility functions for Sarsa(lambda) or ETD(lambda) in PuddleWorld '''

# Choose an action in state epsilon_greedily
# by consulting the action-values given by a linear
# function approximator with weight_vector
def epsilon_greedy(state, weight_vector, epsilon):
    estimated_action_values = []
    # Get apprximate action-values for all legal action in state
    for action in range(num_actions):
        ft = feature(state, action)
        estimated_action_value = np.dot(weight_vector, ft)
        estimated_action_values.append(estimated_action_value)

    if np.random.random() > epsilon:
        max_q = max(estimated_action_values)
        greedy_actions = [action for action, q in enumerate(estimated_action_values) if q == max_q]
        action = np.random.choice(greedy_actions) if len(greedy_actions) != 0 else 0 # if weight_vector overflows and thus max_q = nan we always return 0
        return action 
    else:
        return np.random.choice(list(range(num_actions))) # randomly select an action from [0, 1, 2, 3]

# Check for divergence (when step size is too large so that weight vector becomes inf or nan)
def is_weight_valid(weight_vector):
    if np.isinf(weight_vector).any() or np.isnan(weight_vector).any():
        print('weight vector has become invalid (overflow). Break out and finish the program \n', file=sys.stderr)
        return False
    else:
        return True
