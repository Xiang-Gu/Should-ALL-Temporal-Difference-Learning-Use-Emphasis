#!/lusr/bin/python3
import sys
sys.path.append("..")
from Environment.FiftyStates_1 import FiftyStates_1
from Environment.FiftyStates_2 import FiftyStates_2
import argparse
import numpy as np
from time import sleep

def current_estimated_values(w):
    return [w * i for i in range(1,51)]


parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--case', default='1', type=int,
                    help='which case do you want (1 or 2)?')
parser.add_argument('--filename', default='TD_results', type=str,
                    help='Data/lambda/alpha/filename')
parser.add_argument('--lam', default='0.0', type=float,
                    help='bootstrapping degree')
parser.add_argument('--alpha_2RaisedTo', default='-14', type=float, # NOTICE: for convenince, the input for step size is the exponent of 2.
                    help='step size = 2 ** input!!!')
parser.add_argument('--num_episodes', default='3000', type=int,
                    help='number of training episodes')


args = parser.parse_args()
case = args.case
filename = args.filename
lam = args.lam
alpha = pow(2, args.alpha_2RaisedTo)  # initialize algorithm setting
max_episodes = args.num_episodes


def main():
    w = -10. # parameter of the (non-linear) function approximator; a single scaler
    env = FiftyStates_1() if case == 1 else FiftyStates_2()
    ws = []
    vs = []

    for episode in range(max_episodes):
        print('\n\nAt the begining of ' + str(episode) + 'th episode: w = ' + str(w))
        ws.append(w)
        vs.append(current_estimated_values(w))

        state = env.reset()
        eligibility_trace = 0.
        done = False

        step = 0

        while not done:
            next_state, reward, done = env.step()

            # Update eligibility trace
            eligibility_trace = lam * eligibility_trace + state

            # Compute TD error
            if done:
                TD_error = reward - state * w
            else:
                TD_error = reward + next_state * w - state * w

            # Update weight vector
            w += alpha * TD_error * eligibility_trace

            state = next_state

    np.savetxt(filename, ws)

if __name__ == '__main__':
    main()
