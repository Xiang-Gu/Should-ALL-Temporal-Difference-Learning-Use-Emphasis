#!/lusr/bin/python3
import sys
sys.path.append("..")
from Environment.Cartpole import Cartpole, num_actions
from Util.util import degreeToRadian, feature, estimated_action_value, epsilon_greedy, solved
import numpy as np
import argparse
from scipy import pi

# Query this function to get user-defined Interests to each state
# in the mountaincar environment
def interest(state):
    return 1

# Importance sampling ratio at each time step
def rho(state):
    return 1


'''Use ETD(lambda) to do Control on Cartpole'''

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename', default='ETD_results', type=str, help='Data/0.9/2-13/ETD_results')
parser.add_argument('--lam', default='0.9', type=float, help='bootstrapping degree')
# NOTICE: for convenince, the input for step size is the exponent of 2.
parser.add_argument('--alpha_2RaisedTo', default='-13', type=float, help='step size = 2 ** input!!!')
parser.add_argument('--gamma', default='1.0', type=float, help='discount rate')
parser.add_argument('--epsilon', default='0.0', type=float, help='epsilon')
parser.add_argument('--num_episodes', default='3000', type=int, help='training episodes')

# initialize algorithm setting
args = parser.parse_args()
alpha = pow(2, args.alpha_2RaisedTo)  # initialize algorithm setting
lam = args.lam
gamma = args.gamma
epsilon = args.epsilon
max_episodes = args.num_episodes
max_steps_per_episode = 100000

def main():
    dimension = 162 * num_actions # total number of state groups (3 * 6 * 3 * 3 = 162)
    weight_vector = max_steps_per_episode * np.ones(dimension) # Optimistically initialize weight_vector to be the largest value it can achieve in one episodes to encourage exploration
    env = Cartpole()
    performances = [] # Record the steps of cart before falling at each episode (ceilinged by max_steps_per_episode)

    for idx in range(max_episodes):
        # Terminate the training process when the agent has learned to balance the
        # pole for max_steps_per_episode time steps without falling for 10 consecutive episodes
        if solved(performances, max_steps_per_episode, 10): break

        # Still not successful, keep training!
        print(str(idx) + 'th episode')
        steps_per_episode = 0 # steps balanced in this episode

        state = env.reset()
        action = epsilon_greedy(state, weight_vector, epsilon)
        # Initialize followon trace F_0
        F = interest(state)
        # Initialize Emphasis M_0
        M = lam * interest(state) + (1 - lam) * F
        # Initialize eligibility trace e_0
        eligibility_trace = np.zeros(dimension)
        eligibility_trace[feature(state, action)] += rho(state) * M
        done = False

        while not done:
            next_state, reward, done = env.step(action)

            next_action = epsilon_greedy(next_state, weight_vector, epsilon)

            # Compute TD error
            if done:
                TD_error = reward - estimated_action_value(state, action, weight_vector)
            else:
                TD_error = reward + gamma * estimated_action_value(next_state, next_action, weight_vector) - estimated_action_value(state, action, weight_vector)

            # Update weight vector
            weight_vector += alpha * TD_error * eligibility_trace

            # Update Follow-on trace
            F = rho(state) * gamma * F + interest(next_state)
            # Update Emphasis
            M = lam * interest(next_state) + (1 - lam) * F
            # Update eligibility trace
            eligibility_trace = rho(next_state) * gamma * lam * eligibility_trace
            eligibility_trace[feature(next_state, next_action)] += rho(next_state) * M

            state = next_state
            action = next_action

            # Allow max_steps_per_episode steps at most per episode. Force to terminate if it exceeds this number
            steps_per_episode += 1
            if steps_per_episode >= max_steps_per_episode:
                break

        if not done:
            print('Good job. The pole balanced this episode by at least ' + str(max_steps_per_episode) + ' steps!\n')
        else:
            print('The pole failed in this episode after ' + str(steps_per_episode) + ' steps!\n')
        performances.append(steps_per_episode)

    np.savetxt(args.filename, performances)

if __name__ == '__main__':
    main()
