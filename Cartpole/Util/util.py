import sys
sys.path.append("..")
from Environment.Cartpole import Cartpole, num_actions
from scipy import pi
import numpy as np

# Convert degree to radian
def degreeToRadian(theta):
    return theta * (2 * pi) / 360


def feature(state, action):
    '''
    In Cartpole problem, simply use state aggregation to construct features of state action pair.
    It returns a single integer -- the index of active state-action group.
    state = [x, theta, dx, dtheta]
    Slice first dimension into three intervals at [-2.4, -0.8, 0.8, 2.4] (a state with position less than -2.4 or larger than 2.4 is a invalid input)
    Slice second dimension into six internvals at [-12, -6, -1, 0, 1, 6, 12] (a state with theta less than -12 degree or larger than 12 degree is a invalid input)
    Slice thrid dimension into three intervals at [-inf, -0.5, 0.5, inf]
    Slice fourth dimension into three intervals at [-inf, -50, 50, inf]
    '''
    assert action in list(range(num_actions)), 'Input action is invalid'

    x, theta, dx, dtheta = state
    a = action
    if x < -Cartpole.MAX_TRACK_LENGTH \
        or x > Cartpole.MAX_TRACK_LENGTH \
        or theta < -Cartpole.MAX_THETA \
        or theta > Cartpole.MAX_THETA:
        return -1 # Failure signal

    box = 0
    if x < -0.8:                     box = 0
    elif x < 0.8:                    box = 1
    else:                            box = 2

    if theta < degreeToRadian(-6):   box += 0
    elif theta < degreeToRadian(-1): box += 3
    elif theta < degreeToRadian(0):  box += 6
    elif theta < degreeToRadian(1):  box += 9
    elif theta < degreeToRadian(6):  box += 12
    else:                            box += 15

    if dx < -0.5:                    box += 0
    elif dx < 0.5:                   box += 18
    else:                            box += 36

    if dtheta < -50:                 box += 0
    elif dtheta < 50:                box += 54
    else:                            box += 108

    return box if action == 0 else box + 162 # [0, 161] are for state-action (s, 0) and [162, 323] are for (s, 1)


# Return estimated action value at (state, action) using linear function approximator
def estimated_action_value(state, action, weight_vector):
    ft = feature(state, action)
    return weight_vector[ft] if ft != -1 else 0. # ft = -1 means state is a terminal/failure state so the estimated action value ought to be 0.


# Choose an action in state epsilon_greedily
# by consulting the action-values given by a linear
# function approximator with weight_vector
def epsilon_greedy(state, weight_vector, epsilon):
    estimated_action_values = []
    # Get apprximate action-values for all legal action in state
    for action in range(num_actions):
        estimated_action_values.append(estimated_action_value(state, action, weight_vector))

    if np.random.random() > epsilon: # Greedy action
        action = estimated_action_values.index(max(estimated_action_values))
        return action
    else:
        return np.random.choice(list(range(num_actions))) # random action


# Return true if the past cutOff episodes terminate when the
# pole balanced max_steps_per_episode time steps without falling
def solved(performances, max_steps_per_episode, cutOff):
    if len(performances) >= cutOff and performances[-cutOff:] == [max_steps_per_episode] * cutOff:
        return True
    else:
        return False
