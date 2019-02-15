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
    cut_off_num = 2970
    interval = 30

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

# Extract lambda from the input directory name str
# e.g. directory = TD/Data\\0.8\\
def get_lam(directory):
    result = ''
    flag = False
    for char in reversed(directory):
        if (char == '\\' or char == '/') and not flag:
            flag = True
        elif (char == '\\' or char == '/') and flag:
            break
        elif flag:
            result += char

    return float(result[::-1]) # Reverse the result


def main():
    INF = 99
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--curve', default='FP', type=str,
                    help='Final Performance (FP) or Area Under the Curve (AUC)')
    parser.add_argument('--method', default='TD', type=str,
                    help='method to plot')


    args = parser.parse_args()
    method = args.method
    curve = args.curve
    if curve == 'FP':
        curve_verbose = 'Final Performance Curve'
    elif curve == 'AUC':
        curve_verbose = 'Area Under the Curve'

    if method == 'TD':
        path = 'TD/Data/'
    elif method == 'ETD':
        path = 'ETD/Data/'
    else:
        assert False

    list_directories = glob.glob(path + '*/') # get a list of all directoies for different step sizes

    # Loop for each lambda and plot it
    for each_lam_directory in list_directories:

        print(each_lam_directory)

        # Extract lambda from directory path
        lam = get_lam(each_lam_directory)

        print(lam)

        # Loop for each step size directory inside this lambda and compute their mean and SEM
        list_step_size_directories = glob.glob(each_lam_directory + '*/')
        # result is a list of lists where each list contain [step_size, mean, SEM] inside this lambda directory
        result = []
        # Loop through each step size and compute the mean and SEM
        for each_step_size in list_step_size_directories:
            step_size = get_step_size(each_step_size)
            mean, SEM = compute_mean_and_SEM(each_step_size, curve)
            result.append([step_size, mean, SEM])

        # Sort the results of both method according to the step size
        result.sort(key=lambda x : x[0])

        # modify the data so that inf or nan will be expressed
        # as a very large number in order to plot it
        # for each_point in result:
        #     if np.isinf(each_point[1]) or np.isnan(each_point[1]) or each_point[1] > INF:
        #         each_point[1] = INF
        #         each_point[2] = np.nan
        #         break

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
