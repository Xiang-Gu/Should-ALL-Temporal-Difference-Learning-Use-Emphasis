import numpy as np

'''A simple 50-states Markov Reward Process. We always start from
state 50 and deterministically transition to the previous state -- when
you're at state i you will transition to state (i-1) with reward g(i).
The episode terminates after transition from state 1 to state 0 with reward g(1).
In other words, state 0 is the terminal state. No discounting is used.'''

class FiftyStates_2:
    def __init__(self):
        self._state = 50 # Start with S_0

    def step(self):
        self._state -= 1 # state deterministically transition to the previous one
        done = self._state == 0 # episode terminates when the agent transitions from state 1 to state 0
        # g(50) = -49 and g(i) = 1 for all i != 50
        reward = -49 if self._state == 49 else 1

        return self._state, reward, done

    def reset(self):
        self.__init__()
        return self._state

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state
        return self._state
