# The definition of a stochastic policy for the GridWorld environment

import Deterministic_GridWorld
import numpy as np
import random

class Stochastic_Policy:
    def __init__(self):
        self._probability_action0 = 0.25 # The probability of selecting action 0 ('up')
        self._probability_action1 = 0.25
        self._probability_action2 = 0.25
        self._probability_action3 = 0.25
        self._probabilities = [self._probability_action0, self._probability_action1, self._probability_action2, self._probability_action3]

    def select_action(self, state):
        return np.random.choice([0, 1, 2, 3], p=self._probabilities) # Select an action (0, or 1, or 2, or 3) according to the assigned probability

    def epsilon_greedy(self, Q_state_action, epsilon):
        if random.uniform(0,1) < epsilon: # Take random action
            return np.random.choice([0, 1, 2, 3])
        else: # Take greedy action (if there is multiple greedy actions whose values are the same, return the first action)
            greedy_action = Q_state_action.tolist().index(max(Q_state_action))
            return greedy_action
    

# Extract optimal actions from Q-values
def extract_optimal_actions(Q):
    optimal_actions = np.zeros([Q.shape[0], Q.shape[1]])
    for row in range(Q.shape[0]):
        for col in range(Q.shape[1]):
            action_values = Q[row][col] # Get action-values for (row, col) state
            optimal_action = action_values.tolist().index(max(action_values)) # Find the optimal action w.r.t. the action-values
            optimal_actions[row][col] = optimal_action # Assign this optimal action to corresponding entry in the result
    return optimal_actions




