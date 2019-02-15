import numpy as np

''' A continuing Markov reward process: Three states S_0, S_1, S_2;
In each state, 1/2 probablity of transitioning to itself and 1/2 to the previous state;
Zero reward for each transition'''
class ThreeStates:
    def __init__(self):
        self._state = 0 # Start with S_0

    def step(self):
        if np.random.random() > 0.5: # transition to the next state
            self._state = (self._state - 1) % 3
        else:
            self._state = self._state # transition to itself

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
