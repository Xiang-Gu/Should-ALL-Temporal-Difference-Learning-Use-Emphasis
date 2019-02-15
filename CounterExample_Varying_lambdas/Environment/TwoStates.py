import numpy as np

''' A continuing Markov reward process with varying lambdas: Two states S_0, S_1
One state transitions to the other state and vice versa; Zero reward for each transition'''
class TwoStates:
    def __init__(self):
        self._state = 0 # Start with S_0

    def step(self):
        self._state = (self._state + 1) % 2

        # All transition has a zero reward
        reward = 0
        done = False

        return self._state, reward, done

    def reset(self):
        self.__init__()
        return self._state

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state
        return self._state
