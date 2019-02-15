# The definition for GridWorld -- the environment that the agent will continuously interact with

import numpy as np

class GridWorld:
    def __init__(self):
        self._num_rows = 5
        self._num_cols = 5
        self._state = (0, 0)
        self._t = 0
        self._terminal_states = {(self._num_rows - 1, self._num_cols - 1)} # A list of terminal states
        self._actions = [0, 1, 2, 3] # 0, 1, 2, 3 represents 'up', 'right', 'down', 'left' respectively

    # Interface methods
    def step(self, action):
        self._state = self._get_next_state(self._state, action)
        reward = self._reward(self._state)
        done = self._state in self._terminal_states # The termination state is at the top-right corner of the grid world
        self._t += 1
        return self._state, reward, done
    
    def reset(self):
        self._state = (0, 0)
        self._t = 0
        return self._state

    def get_num_rows(self):
        return self._num_rows
    
    def get_num_cols(self):
        return self._num_cols
    
    def get_actions(self):
        return self._actions

    # Helper method for implementing the functions inside the class
    def _get_next_state(self, current_state, action):
        if action == 0:
            if self._state[0] == self._num_rows - 1:
                return current_state
            else:
                return (current_state[0] + 1, current_state[1])
        elif action == 1:
            if self._state[1] == self._num_cols - 1:
                return current_state
            else:
                return (current_state[0], current_state[1] + 1)
        elif action == 2:
            if self._state[0] == 0:
                return current_state
            else:
                return (current_state[0] - 1, current_state[1])
        else:
            if self._state[1] == 0:
                return current_state
            else:
                return (current_state[0], current_state[1] - 1)
    
    def _reward(self, state):
        return -1 if state in self._terminal_states else -1
        





        