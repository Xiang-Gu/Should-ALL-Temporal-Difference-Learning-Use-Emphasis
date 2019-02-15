#! /usr/bin/env python

from __future__ import print_function

import os
import subprocess
import argparse
import getpass

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--method', default='TD', type=str,
                    help='Which method do you want (TD or ETD)?')
parser.add_argument('--lam', default='1.0', type=float,
                    help='bootstrapping degree')
parser.add_argument('--alpha_2RaisedTo', default='-8', type=int, # NOTICE: for convenince, the input for step size is the exponent of 2.
                    help='step size = 2 ** input!!!')
parser.add_argument('--gamma', default='1.0', type=float,
                    help='discount rate')
parser.add_argument('--epsilon', default='0.01', type=float,
                    help='exploration rate')
parser.add_argument('--num_episodes', default='1000', type=int,
                    help='training episodes')
parser.add_argument('--num_trials', default='30', type=int,
                    help='number of repeating trials')


def submit_to_condor(executable_file, outfile, lam, alpha_2RaisedTo, gamma, epsilon, num_episodes, num_trials):
    argument = ('--filename=%s --lam=%s --alpha_2RaisedTo=%s --gamma=%s --epsilon=%s --num_episodes=%s'
                % (outfile, lam, alpha_2RaisedTo, gamma, epsilon, num_episodes))

    submit_file = 'Universe = vanilla\n'
    submit_file += 'Executable = ' + executable_file + "\n"
    submit_file += 'Arguments = ' + argument + '\n'
    submit_file += 'Log = /dev/null\n'
    submit_file += 'Error = ' + outfile + '.err\n'
    submit_file += '+Group = "GRAD"\n' + '+Project = "AI_ROBOTICS"\n'
    submit_file += '+ProjectDescription = "MountainCar_Control"\n'

    for _ in range(num_trials):
        submit_file += 'Queue\n'


    proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE)
    proc.stdin.write(submit_file.encode())
    proc.stdin.close()


def main():
    # Read arguments
    args = parser.parse_args()
    lam = args.lam
    alpha_2RaisedTo = args.alpha_2RaisedTo
    gamma = args.gamma
    epsilon = args.epsilon
    num_episodes = args.num_episodes
    num_trials= args.num_trials
    executable_file = '/u/xiangu/Desktop/UofA/MountainCar_Control/'
    outfile = '/projects/agents2/villasim/xiangu/MountainCar_Control/'
    if args.method == 'TD':
        executable_file += 'TD/TD.py'
        outfile += 'TD/Data/' + str(lam) + '/2' + str(alpha_2RaisedTo) + '/TD_results_$(Process)'
    elif args.method == 'ETD':
        executable_file += 'ETD/ETD.py'
        outfile += 'ETD/Data/' + str(lam) + '/2' + str(alpha_2RaisedTo) + '/ETD_results_$(Process)'
    else:
        print('Unrecognizable Method: ' + str(args.method) + '\n Please enter either TD or ETD')


    submit_to_condor(executable_file, outfile, lam, alpha_2RaisedTo, gamma, epsilon, num_episodes, num_trials)


if __name__ == "__main__":
    main()
