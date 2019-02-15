from matplotlib import pyplot as plt
import argparse
import numpy as np
import glob
import math
import os

# Compute the mean and SEM of files in directory
# curve can be FP (Final Performance) or AUC (Area Under the Curve)
# directory should be one of step size directory (e.g. TD/Data/0.0/2-3/)
def compute_mean_and_SEM(directory, curve):
    list_files = glob.glob(directory + '*') # get a list of all files in onpolicy_results_directory
    num_files = len(list_files) # Number of files (# of trials we ran) in the directory

    # Collect data from cut_off_num to (cut_off_num + interval) for plot FP curve
    cut_off_num = 49950
    interval = 50

    means_each_file = [] # means_each_file stores the mean of last 0.1% samples in each file
    mean = 0.0
    SEM = 0.0

    for idx in range(num_files):
        file_name = list_files[idx]
        if not file_name.endswith('.err'):
            data = [] # It stores samples in this file

            file = open(file_name, 'r')
            fline = file.readlines()
            num_line = 0 # Used for FP curve only

            for eachline in fline:
                if curve == 'FP': # Pick only data within [cut_off_num, cut_off_num + interval] episodes
                    if num_line >= (cut_off_num + interval):
                        break
                    if num_line >= cut_off_num:
                        data.append(float(eachline))
                    num_line += 1
                elif curve == 'AUC': # Add all episodes
                    data.append(float(eachline))
                else:
                    assert False
            means_each_file.append(np.mean(data))

    mean = np.mean(means_each_file)
    SEM = np.std(means_each_file)
    return mean, SEM

# Extract step size (a float number) from a path
# e.g. TD/Data/0.0\\2-9\\  -->  2**(-9) = 0.001953125
def get_step_size(each_alpha_directory):
    result = ''
    flag = False
    for char in reversed(each_alpha_directory):
        if (char == '\\' or char == '/') and not flag:
            flag = True
        elif (char == '\\' or char == '/') and flag:
            break
        elif flag:
            result += char
    result = result[::-1] # Reverse the string (e.g. 01-2 --> 2-10)

    base = result[0]
    exp = result[1:]

    return pow(float(base), float(exp))


def main():
    INF = 99
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lam', default=0.0, type=float,
                        help='compare both methods with this lambda')
    parser.add_argument('--curve', default='FP', type=str,
                        help='method to plot (either FP or AUC)')



    args = parser.parse_args()
    lam = args.lam
    curve = args.curve
    if curve == 'FP':
        curve_verbose = 'Final Performance Curve'
    elif curve == 'AUC':
        curve_verbose = 'Area Under the Curve'


    TD_common_path = '~/Desktop/Research_UofA/MountainCar/TD/Data/' + str(lam) + '/'
    ETD_common_path = '~/Desktop/Research_UofA/MountainCar/ETD/Data/' + str(lam) + '/'

    list_TD_directories = glob.glob(TD_common_path + '*/') # get a list of all directoies for different step sizes
    list_ETD_directories = glob.glob(ETD_common_path + '*/')

    # result_TD is a list of lists where each list contains
    # [step_size, mean, SEM]
    result_TD = []
    result_ETD = []

    for each_alpha_directory_TD, each_alpha_directory_ETD in zip(list_TD_directories, list_ETD_directories):
        # Extract step size of current directory for TD method
        step_size_TD = get_step_size(each_alpha_directory_TD)
        # Compute mean and SEM (of final performance) in this directory
        mean_TD, SEM_TD = compute_mean_and_SEM(each_alpha_directory_TD, curve)
        # Add this data point [step_size, mean, SEM] to results
        result_TD.append([step_size_TD, mean_TD, SEM_TD])


        # Do the same thing for each ETD alpha directory
        step_size_ETD = get_step_size(each_alpha_directory_ETD)
        mean_ETD, SEM_ETD = compute_mean_and_SEM(each_alpha_directory_ETD, curve)
        result_ETD.append([step_size_ETD, mean_ETD, SEM_ETD])

    # Sort the results of both method according to the step size
    result_TD.sort(key=lambda x : x[0])
    result_ETD.sort(key=lambda x : x[0])

    # modify the data so that inf or nan will be expressed
    # as a very large number in order to plot it
    for each_point in result_TD:
        if np.isinf(each_point[1]) or np.isnan(each_point[1]) or each_point[1] > INF:
            each_point[1] = INF
            each_point[2] = np.nan
            break
    for each_point in result_ETD:
        if np.isinf(each_point[1]) or np.isnan(each_point[1]) or each_point[1] > INF:
            each_point[1] = INF
            each_point[2] = np.nan
            break

    # Start to plot the result
    alphas_TD = [each_point[0] for each_point in result_TD]
    averages_TD = [each_point[1] for each_point in result_TD]
    errors_TD = [each_point[2] for each_point in result_TD]

    alphas_ETD = [each_point[0] for each_point in result_ETD]
    averages_ETD = [each_point[1] for each_point in result_ETD]
    errors_ETD = [each_point[2] for each_point in result_ETD]

    # Plot on differnt scale of x-axis
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xscale('log')
    ax1.set_xlim(left=1.e-6, right=1.e-2)
    ax1.set_ylim(bottom=20, top=30)
    ax1.set_yticklabels([''])
    ax1.set_xlabel('Step Sizes', fontsize=35)  # we already handled the x-label with ax2
    # ax1.set_ylabel(r'$\sqrt{\widehat{\overline{VE}}}$', rotation=0, fontsize=25, labelpad=40)
    ax1.errorbar(alphas_ETD, averages_ETD, yerr=errors_ETD, fmt='--o', label='ETD', color=color)
    ax1.tick_params(axis='x', labelcolor=color, labelsize=25)
    ax1.tick_params(axis='y', labelsize=30)
    ax2 = ax1.twiny()  # instantiate a second axes that shares the same y-axis

    color = 'tab:blue'
    ax2.set_xscale('log')
    ax2.set_xlim(left=5.e-6, right=1.e-1)
    ax2.set_ylim(bottom=20, top=30)
    ax2.errorbar(alphas_TD, averages_TD, yerr=errors_TD, fmt='-o', label='TD', color=color)
    ax2.tick_params(axis='x', labelcolor=color, labelsize=23)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines + lines2, labels + labels2, loc=0, fontsize=15)
    # plt.title(r'$\lambda$' + ' = ' + str(lam), fontsize=25)

    # Plot with the same scale of x-axis
    # plt.title(curve_verbose +'\n' + r'$\lambda$' + ' = ' + str(lam), fontsize=15)
    # plt.xscale('log')
    # plt.xlabel(r'$\alpha$', fontsize=20)
    # plt.ylabel(r'$\sqrt{\overline{VE}}$', rotation=0)
    # plt.errorbar(alphas_TD, averages_TD, yerr=errors_TD, fmt='-o', label='TD')
    # plt.errorbar(alphas_ETD, averages_ETD, yerr=errors_ETD, fmt='--o', label='ETD')
    # plt.legend(loc='best', fontsize=15)
    fig.savefig('/Users/rlai/Desktop/test2.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
