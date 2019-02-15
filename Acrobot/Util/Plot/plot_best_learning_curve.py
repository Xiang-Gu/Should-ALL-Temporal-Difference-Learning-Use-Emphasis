from matplotlib import pyplot as plt
import argparse
import numpy as np
import math
from plot_util import get_all_files, get_mean_and_SEM_all_stepSizes, COMMON_PATH

'''
Plot the best learning curve of TD and ETD method for a certain given lambda
The whole program is divided into three parts:
    1. Compute the performance for all step sizes for both methods;
    2. Find the best step size that gives the best performance;
    3. Plot the learning curve of that step size.
'''
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lam', default=0.0, type=float,
                    help='compare both methods with this lambda')
parser.add_argument('--curve', default='FP', type=str,
                    help='method to plot (either FP or AUC)')

args = parser.parse_args()
lam = args.lam
curve = args.curve
curve_verbose = 'Final Performance Curve' if curve == 'FP' else 'Area Under the Curve'

def main():
    TD_path = COMMON_PATH + 'TD/Data/' + str(lam) + '/'
    ETD_path = COMMON_PATH + 'ETD/Data/' + str(lam) + '/'

    # result_TD is a list of lists where each component list is
    # [step_size, mean, SEM]
    result_TD = get_mean_and_SEM_all_stepSizes(TD_path, curve)
    result_ETD = get_mean_and_SEM_all_stepSizes(ETD_path, curve)

    # Sort the results of both method according to the mean performance measure
    result_TD.sort(key=lambda x : x[1])
    result_ETD.sort(key=lambda x : x[1])

    best_TD_step_size = result_TD[0][0]
    best_TD_step_size_2RaisedTo = int(math.log(best_TD_step_size, 2))
    best_ETD_step_size = result_ETD[0][0]
    best_ETD_step_size_2RaisedTo = int(math.log(best_ETD_step_size, 2))

    # Now that we have the best step size, let's go and plot the learning curve of that step size
    best_TD_directory = COMMON_PATH + 'TD/Data/' + str(lam) + '/2' + str(best_TD_step_size_2RaisedTo) + '/'
    best_ETD_directory = COMMON_PATH + 'ETD/Data/' + str(lam) + '/2' + str(best_ETD_step_size_2RaisedTo) + '/'

    # measures_TD is a list of lists, where each element (each list)
    # is the measures_TD of a trial.
    measures_TD = get_all_files(best_TD_directory)
    measures_ETD = get_all_files(best_ETD_directory)

    # Plot the graph
    mean_TD = np.mean(measures_TD, axis=0)
    yerror_TD = np.std(measures_TD, axis=0) / math.sqrt(len(measures_TD)) # Approximate the standard error of the mean using std(X)/sqrt(n), where n = len(X)
    mean_ETD = np.mean(measures_ETD, axis=0)
    yerror_ETD = np.std(measures_ETD, axis=0) / math.sqrt(len(measures_ETD))

    # plt.yscale('log')
    plt.errorbar(np.arange(mean_TD.size), mean_TD, yerr=yerror_TD, label=r'TD, $\alpha$=' + str(best_TD_step_size))
    plt.errorbar(np.arange(mean_ETD.size), mean_ETD, yerr=yerror_ETD, label=r'ETD, $\alpha$=' + str(best_ETD_step_size))
    plt.xlabel('Episodes')
    plt.ylabel('Steps per Episode\nAveraged over ' + str(len(measures_TD)) + ' runs', rotation=0)
    # plt.title('Best learning curve for TD and ETD Acrobot control \n' + r'$\lambda$ = ' + str(lam) + '\n' + str(curve_verbose) )
    plt.title(r'$\lambda$ = ' + str(lam), fontsize=25)
    plt.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    main()
