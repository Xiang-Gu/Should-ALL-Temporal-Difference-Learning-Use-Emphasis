from matplotlib import pyplot as plt
import argparse
import numpy as np
import glob
import math
import os
from plot_util import INF, get_method_lam_alpha, compute_mean_and_SEM, COMMON_PATH, get_mean_and_SEM_all_stepSizes

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--curve', default='FP', type=str,
                help='Final Performance (FP) or Area Under the Curve (AUC)')
parser.add_argument('--method', default='TD', type=str,
                help='method to plot')
parser.add_argument('--performance_measure', default='step', type=str,
                help='performance measure (either step or reward)')

args = parser.parse_args()
method = args.method
performance_measure = args.performance_measure
curve = args.curve
curve_verbose = 'Final Performance Curve' if curve == 'FP' else 'Area Under the Curve'
path = COMMON_PATH + 'TD/Data/' if method == 'TD' else COMMON_PATH + 'ETD/Data/'

def main():
    list_directories = glob.glob(path + '*/') # get a list of all directoies for different step sizes

    # Loop for each lambda and plot it
    for each_lam_directory in list_directories:
        print(each_lam_directory)

        _, lam, _ = get_method_lam_alpha(each_lam_directory)
        result = get_mean_and_SEM_all_stepSizes(each_lam_directory, curve, performance_measure)
        # Sort the result according to the step size
        result.sort(key=lambda x : x[0])

        alphas = [each_point[0] for each_point in result]
        averages = [each_point[1] for each_point in result]
        errors = [each_point[2] for each_point in result]
        # # Plot the curve for current lambda
        plt.errorbar(alphas, averages, yerr=errors, fmt='-o', label=r'$\lambda$'+ '= ' + str(lam))

    plt.title(method + '\n' + curve_verbose, fontsize=15)
    plt.xscale('log')
    plt.xlabel(r'$\alpha$', fontsize=20)
    plt.ylabel('Steps per episode', rotation=0)
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
