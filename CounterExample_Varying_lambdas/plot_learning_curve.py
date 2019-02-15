from matplotlib import pyplot as plt
import argparse
import numpy as np
import glob
import math

'''Plot the learning curve (change of parameter) of TD and ETD methods on the same plot '''

# Read data from a directory and plot the results
def main():
    TD_filename = 'TD/Data/TD_results_weights'
    ETD_filename = 'ETD/Data/ETD_results_weights'
    # TD_filename = 'TD/Data/TD_results_estimated_values'
    # ETD_filename = 'ETD/Data/ETD_results_estimated_values'

    parameters_one_TD = []
    parameters_two_TD = []
    parameters_one_ETD = []
    parameters_two_ETD = []

    # Load data from TD results (the problem is deterministic so there is only one file)
    file = open(TD_filename, 'r')
    fline = file.readlines()
    # Add each line (which is one float) to Measures
    for eachline in fline:
        parameters = eachline.split(' ')
        parameters_one_TD.append(float(parameters[0]))
        parameters_two_TD.append(float(parameters[1]))

    # Load data from ETD results (the problem is deterministic so there is only one file)
    file = open(ETD_filename, 'r')
    fline = file.readlines()
    # Add each line (which is one float) to Measures
    for eachline in fline:
        parameters = eachline.split(' ')
        parameters_one_ETD.append(float(parameters[0]))
        parameters_two_ETD.append(float(parameters[1]))

    # Plot the graph
    # plt.title('Weights Change', fontsize=25)
    fig, ax = plt.subplots(nrows=1,ncols=1)
    plt.xticks([0, 5000, 10000, 15000], ['0', '5K', '10K', '15K'])
    plt.yticks([-100000, -50000, 0, 50000], ['-100K', '-50K', '0', '50K'])
    ax.plot(np.arange(len(parameters_one_TD)), parameters_one_TD, label='TD: ' + r'$w_1$', color='tab:blue')
    ax.plot(np.arange(len(parameters_two_TD)), parameters_two_TD, label='TD: ' + r'$w_2$', color='tab:blue')
    ax.plot(np.arange(len(parameters_one_ETD)), parameters_one_ETD, label='ETD: ' + r'$w_1$', color='tab:red')
    ax.plot(np.arange(len(parameters_two_ETD)), parameters_two_ETD, label='ETD: ' + r'$w_2$', color='tab:red')
    ax.set_xlabel('Iterations', fontsize=25)
    ax.set_ylabel('Weights', fontsize=25, rotation=0, labelpad=10)
    ax.set_xlim(left=0, right=15000)
    ax.set_ylim(bottom=-100000, top=80000)
    ax.tick_params(labelsize=23, which='major', axis='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('/Users/rlai/Desktop/test5.pdf', format='pdf', dpi=300, bbox_inches='tight')
    # plt.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    main()
