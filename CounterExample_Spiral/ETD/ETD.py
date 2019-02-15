#! /usr/bin/env python
import sys
sys.path.append("..")
from Environment.ThreeStates import ThreeStates
import numpy as np
from scipy.linalg import expm
import os
import argparse
from time import sleep
from copy import deepcopy

epsilon = 0.05
Q = np.array([[1., .5, 1.5], [1.5, 1., .5], [.5, 1.5, 1.]])
A = Q + epsilon * np.eye(3)
# V_init = np.array([[1], [1], [-2]])
V_init = np.array([[10], [10], [-20]])

# A nonlinear (state values) function approximator
def value_function_approximator(state, w):
    assert state in {0, 1, 2}, 'state out of range'
    V = np.matmul(expm(A * w), V_init) # V(w) = e^{Aw} * V(0), where A = Q + \epsilon I
    return float(V[state])

# Derivative of the function approximator of state w.r.t to theta
# It returns a scaler value which is the derivative evaluated at theta
def gradient_V(state, w):
    grad_V = np.matmul(A, np.matmul(expm(A * w), V_init)) # dV/dw = A * V(w)
    return float(grad_V[state])

# Query this function to get user-defined Interests to each state
# in the mountaincar environment
def interest(state):
    return 1

# Importance sampling ratio at each time step
def rho(state):
    return 1


# Use ETD to do prediction on this contrived three-state MDP
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename', default='ETD_results', type=str,
                    help='Data/lambda/alpha/filename')
parser.add_argument('--lam', default='1.0', type=float,
                    help='bootstrapping degree')
parser.add_argument('--alpha_2RaisedTo', default='-8', type=float, # NOTICE: for convenince, the input for step size is the exponent of 2.
                    help='step size = 2 ** input!!!')
parser.add_argument('--gamma', default='1.0', type=float,
                    help='discount rate')
parser.add_argument('--num_trainingsteps', default='3000', type=int,
                    help='training steps')


args = parser.parse_args()
filename = args.filename
lam = args.lam
alpha = pow(2, args.alpha_2RaisedTo)  # initialize algorithm setting
gamma = args.gamma
max_steps = args.num_trainingsteps


def main():
    w = -10. # parameter of the (non-linear) function approximator; a single scaler
    env = ThreeStates()
    state = env.reset()

    # Initialize follow-on trace F_0
    F = interest(state)
    # Initialize Emphasis M_0
    M = lam * interest(state) + (1 - lam) * F
    # Initialize eligibility trace e_0
    eligibility_trace = rho(state) * M * gradient_V(state, w)

    step = 0 # A counter of training process. Terminate when it exceeds max_steps
    ws = [w] # Record the parameter theta at each step. Add the first theta (0) to it
    vs = [[value_function_approximator(0, w), value_function_approximator(1, w), value_function_approximator(2, w)]]


    while step < max_steps:
        print('\n' + str(step) + 'th step: w = ' + str(w))
        print('estimated value for three states = (' + str(value_function_approximator(0, w)) + ', ' + str(value_function_approximator(1, w)) + ', ' + str(value_function_approximator(2, w)) + ')')
        # sleep(0.01)

        next_state, reward, _ = env.step()

        # Compute TD error
        TD_error = reward + gamma * value_function_approximator(next_state, w) - value_function_approximator(state, w)

        # Update weight vector
        w += alpha * TD_error * eligibility_trace

        # Update Follow-on trace
        F = rho(state) * gamma * F + interest(next_state)
        # Update Emphasis
        M = lam * interest(next_state) + (1 - lam) * F
        # Update eligibility trace
        eligibility_trace = rho(next_state) * (gamma * lam * eligibility_trace + M * gradient_V(state, w))

        state = next_state

        ws.append(w)
        vs.append([value_function_approximator(0, w), value_function_approximator(1, w), value_function_approximator(2, w)])
        step += 1

        # Dynamic Programmin style update
        # print('\n' + str(step) + 'th step: w = ' + str(w))
        # print('estimated value for three states = (' + str(value_function_approximator(0, w)) + ', ' + str(value_function_approximator(1, w)) + ', ' + str(value_function_approximator(2, w)) + ')')
        #
        # state = step % 3
        # # Transition back to itself
        # TD_error_1 = 0 + gamma * value_function_approximator(state, w) - value_function_approximator(state, w)
        # # Transition to next state
        # TD_error_2 = 0 + gamma * value_function_approximator((state-1) % 3, w) - value_function_approximator(state, w)
        # # Expected error
        # TD_error = 0.5 * (TD_error_1 + TD_error_2)
        #
        # w += alpha * TD_error * eligibility_trace
        #
        # F = gamma * F + 1
        # eligibility_trace = F * gradient_V((state+1) % 3, w)
        #
        # ws.append(w)
        # vs.append([value_function_approximator(0, w), value_function_approximator(1, w), value_function_approximator(2, w)])
        # step += 1

    np.savetxt(filename + '_weights', ws)
    np.savetxt(filename + '_values', vs)

if __name__ == '__main__':
    main()
