#!/lusr/bin/python3
import sys
sys.path.append("..")
from TileCoder.tilecoder import IHT, tileswrap
from Environment.Acrobot import Acrobot, num_actions
from Util.util import is_weight_valid
import numpy as np
import argparse
from time import sleep
from scipy import pi
from copy import deepcopy

# Use Rich's tile coder software to construct binary features for state-action pair in this Acrobot problem
maxSize = 100800 # (12 * 7**4 + 12 * 7**3 + 12 * 7**2 + 12 * 7) * 3 = 100800 tiles in total
iht = IHT(maxSize) # Probably don't need that many since we have in totaly 48 tilings

def feature(state, action):
    '''
    Return the feature (of length 48), constructed by tile coder, of a state-action pair in the Acrobot problem
    :param s: a list of four floats representing [theta1, theta2, dtheta1, dtheta2]
    :param a: a integer in {0, 1, 2}
    Reference: Sutton, R. S. (1996). Generalization in reinforcement learning: Successful examples using sparse coarse coding.
               In Advances in neural information processing systems (pp. 1038-1044).
    '''

    s = deepcopy(state) # state is a list so we don't want to modify input state
    a = action
    scaleFactorAngle = 6 / (2 * pi) # theta1, theta2 are in the range [0, 2pi] and we want divide those two dimensions into 6 intervals
    scaleFactorVel1 = 6 / (8 * pi) # dtheta1 is in [-4pi, 4pi] and we want to divide this dimension into 6 intervals
    scaleFactorVel2 = 6 / (18 * pi) # dtheta2 is in [-9pi, 9pi] and we want to divide this dimension into 6 intervals

    # Change the scale of the input variable because this tilecode use integer boundaries
    s[0] *= scaleFactorAngle
    s[1] *= scaleFactorAngle
    s[2] *= scaleFactorVel1
    s[3] *= scaleFactorVel2

    # Tilings on all four dimensions.
    feature1 = tileswrap(iht, 12, [s[0], s[1], s[2], s[3]], [6, 6, False, False], ints=[a])

    # Tilings on three (of the four) dimensions. We have in total 4 such sets.
    feature2 = tileswrap(iht, 3, [s[0], s[1], s[2]], [6, 6, False], ints=[a,0]) \
        + tileswrap(iht, 3, [s[0], s[1], s[3]], [6, 6, False], ints=[a,1]) \
        + tileswrap(iht, 3, [s[0], s[2], s[3]], [6, False, False], ints=[a, 2]) \
        + tileswrap(iht, 3, [s[1], s[2], s[3]], [6, False, False], ints=[a, 3])

    # Tilings on two (of the four) dimensions. We have in total 6 such sets.
    feature3 = tileswrap(iht, 2, [s[0], s[1]], [6, 6], ints=[a, 0]) \
        + tileswrap(iht, 2, [s[0], s[2]], [6, False], ints=[a, 1]) \
        + tileswrap(iht, 2, [s[0], s[3]], [6, False], ints=[a, 2]) \
        + tileswrap(iht, 2, [s[1], s[2]], [6, False], ints=[a, 3]) \
        + tileswrap(iht, 2, [s[1], s[3]], [6, False], ints=[a, 4]) \
        + tileswrap(iht, 2, [s[2], s[3]], [False, False], ints=[a, 5])

    # Tiling on one (of the four) dimensions. We have in total 4 such sets.
    feature4 = tileswrap(iht, 3, [s[0]], [6], ints=[a,0]) \
        + tileswrap(iht, 3, [s[1]], [6], ints=[a,1]) \
        + tileswrap(iht, 3, [s[2]], [False], ints=[a,2]) \
        + tileswrap(iht, 3, [s[3]], [False], ints=[a,3])

    return feature1 + feature2 + feature3 + feature4

# Return estimated action value at (state, action) using linear function approximator
def estimated_action_value(state, action, weight_vector):
    return sum([weight_vector[idx] for idx in feature(state, action)])

# Choose an action in state epsilon_greedily
# by consulting the action-values given by a linear
# function approximator with weight_vector
def epsilon_greedy(state, weight_vector, epsilon):
    estimated_action_values = []
    # Get apprximate action-values for all legal action in state
    for action in range(num_actions):
        estimated_action_values.append(estimated_action_value(state, action, weight_vector))

    if np.random.random() > epsilon: # Greedy action
        max_q = max(estimated_action_values)
        greedy_actions = [action for action, q in enumerate(estimated_action_values) if q == max_q]
        action = np.random.choice(greedy_actions) if len(greedy_actions) != 0 else 0 # if weight_vector overflows and thus max_q = nan we always return 0
        return action
    else:
        return np.random.choice(list(range(num_actions))) # random action


'''Sarsa(lambda) algorithm that optimize the policy for PuddleWorld problem'''

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename', default='TD_results', type=str, help='Data/0.0/2-9/TD_results_0')
parser.add_argument('--lam', default='0.9', type=float, help='bootstrapping degree')
# NOTICE: for convenince, the input for step size is the exponent of 2.
parser.add_argument('--alpha_2RaisedTo', default='-5', type=float, help='step size = 2 ** input!!!')
parser.add_argument('--gamma', default='1.0', type=float, help='discount rate')
parser.add_argument('--epsilon', default='0.0', type=float, help='epsilon')
parser.add_argument('--num_episodes', default='1000', type=int, help='training episodes')

# initialize algorithm setting
args = parser.parse_args()
alpha = pow(2, args.alpha_2RaisedTo)
lam = args.lam
gamma = args.gamma
epsilon = args.epsilon
max_episodes = args.num_episodes
max_steps_per_episode = 999

def main():
    dimension = maxSize # total number of tiles
    weight_vector = np.zeros(dimension)
    env = Acrobot()
    performances = [] # Record the steps used for each episode

    for idx in range(max_episodes):
        # Check for overflow in weight vector (happens when step size is too large)
        if not is_weight_valid(weight_vector):
            # Replace the performance measure in last episode, when the overflow occurs, to infinity as an error signal
            print('Invalid value encountered. Break out!')
            performances[-1] = np.float('inf')
            break

        print(str(idx) + 'th episode')
        steps_per_episode = 0 # steps required in this episode


        state = env.reset()
        action = epsilon_greedy(state, weight_vector, epsilon)
        eligibility_trace = np.zeros(dimension)
        done = False

        while not done:
            next_state, reward, done = env.step(action)

            # Update eligibility trace
            eligibility_trace = gamma * lam * eligibility_trace
            for idx in feature(state, action): # Leverage binary feature + linear function approximation property
                eligibility_trace[idx] += 1

            next_action = epsilon_greedy(next_state, weight_vector, epsilon)

            # Compute TD error
            if done:
                TD_error = reward - estimated_action_value(state, action, weight_vector)
            else:
                TD_error = reward + gamma * estimated_action_value(next_state, next_action, weight_vector) - estimated_action_value(state, action, weight_vector)

            # Update weight vector
            weight_vector += alpha * TD_error * eligibility_trace

            state = next_state
            action = next_action

            # Allow max_steps_per_episode steps at most per episode. Force to terminate if it exceeds this number
            steps_per_episode += 1
            if steps_per_episode >= max_steps_per_episode:
                break


        print('steps used in this episode: ' + str(steps_per_episode) + '\n')
        performances.append(steps_per_episode)

    np.savetxt(args.filename, performances)

if __name__ == '__main__':
    main()
