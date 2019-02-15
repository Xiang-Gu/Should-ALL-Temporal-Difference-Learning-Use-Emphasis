from matplotlib import pyplot as plt
import argparse
import numpy as np
import glob
import math
import os
import plot_learning_curve
import math

# Compute the mean and SEM of files in directory
# curve can be FP (Final Performance) or AUC (Area Under the Curve)
# directory should be one of step size directory (e.g. TD/Data/0.0/2-3/)
def compute_mean_and_SEM(directory, curve):
    list_files_TD = glob.glob(directory + '*') # get a list of all files in onpolicy_results_directory
    num_files = len(list_files_TD) # Number of files (# of trials we ran) in the directory

    # Collect data from cut_off_num to (cut_off_num + interval) for plot FP curve
    cut_off_num = 2970
    interval = 30

    means_each_file = [] # means_each_file stores the mean of last 0.1% samples in each file
    mean = 0.0
    SEM = 0.0

    for idx in range(num_files):
        file_name_TD = list_files_TD[idx]
        if not file_name_TD.endswith('.err'):
            data = [] # It stores samples in this file

            file = open(file_name_TD, 'r')
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
    INF = 99999
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


    TD_common_path = 'TD/Data/' + str(lam) + '/'
    ETD_common_path = 'ETD/Data/' + str(lam) + '/'

    list_TD_directories = glob.glob(TD_common_path + '*/') # get a list of all directoies for different step sizes
    list_ETD_directories = glob.glob(ETD_common_path + '*/')

    # result_TD is a list of lists where each list contains
    # [step_size, mean, SEM]
    result_TD = []
    result_ETD = []

    for each_alpha_directory_TD in list_TD_directories:
        # Extract step size of current directory for TD method
        step_size_TD = get_step_size(each_alpha_directory_TD)
        # Compute mean and SEM (of final performance) in this directory
        mean_TD, SEM_TD = compute_mean_and_SEM(each_alpha_directory_TD, curve)
        # Add this data point [step_size, mean, SEM] to results
        result_TD.append([step_size_TD, mean_TD, SEM_TD])

    for each_alpha_directory_ETD in list_ETD_directories:
        # Do the same thing for each ETD alpha directory
        step_size_ETD = get_step_size(each_alpha_directory_ETD)
        mean_ETD, SEM_ETD = compute_mean_and_SEM(each_alpha_directory_ETD, curve)
        result_ETD.append([step_size_ETD, mean_ETD, SEM_ETD])


    # Sort the results of both method according to the performance measure
    result_TD.sort(key=lambda x : x[1])
    result_ETD.sort(key=lambda x : x[1])

    best_TD_step_size = result_TD[0][0]
    best_TD_step_size_2RaisedTo = int(math.log(best_TD_step_size, 2))
    best_ETD_step_size = result_ETD[0][0]
    best_ETD_step_size_2RaisedTo = int(math.log(best_ETD_step_size, 2))

    best_TD_directory = 'TD/Data/' + str(lam) + '/2' + str(best_TD_step_size_2RaisedTo) + '/'
    best_ETD_directory = 'ETD/Data/' + str(lam) + '/2' + str(best_ETD_step_size_2RaisedTo) + '/'


    list_files_TD = glob.glob(best_TD_directory + '*') # get a list of all files in onpolicy_results_directory
    num_trials_TD = len(list_files_TD) # Number of files (# of trials we ran) in the directory
    list_files_ETD = glob.glob(best_ETD_directory + '*') # get a list of all files in onpolicy_results_directory
    num_trials_ETD = len(list_files_ETD) # Number of files (# of trials we ran) in the directory


    # Measures_TD is a list of lists, where each element (each list)
    # is the Measures_TD of a trial
    Measures_TD = []
    Measures_ETD = []

    # Fill up Measures_TD with differnt runs in best_TD_directory
    for idx in range(num_trials_TD):
        file_name_TD = list_files_TD[idx]
        if not file_name_TD.endswith('.err'):
            Measure_current_trial_TD = []
            file = open(file_name_TD, 'r')
            fline = file.readlines()
            # Add each line (which is one float) to Measures_TD
            for eachline in fline:
                Measure_current_trial_TD.append(float(eachline))
            Measures_TD.append(Measure_current_trial_TD)

    # Fill up Measures_ETD with differnt runs in best_ETD_directory
    for idx in range(num_trials_ETD):
        file_name_ETD = list_files_ETD[idx]
        if not file_name_ETD.endswith('.err'):
            Measure_current_trial_ETD = []
            file = open(file_name_ETD, 'r')
            fline = file.readlines()
            # Add each line (which is one float) to Measures_TD
            for eachline in fline:
                Measure_current_trial_ETD.append(float(eachline))
            Measures_ETD.append(Measure_current_trial_ETD)

    # Plot the graph
    mean_TD = np.mean(Measures_TD, axis=0)
    yerror_TD = np.std(Measures_TD, axis=0) / math.sqrt(len(Measures_TD))
    mean_ETD = np.mean(Measures_ETD, axis=0)
    yerror_ETD = np.std(Measures_ETD, axis=0) / math.sqrt(len(Measures_ETD))
    # plt.yscale('log')
    plt.errorbar(np.arange(mean_TD.size), mean_TD, yerr=yerror_TD, label='TD, alpha=' + str(best_TD_step_size))
    plt.errorbar(np.arange(mean_ETD.size), mean_ETD, yerr=yerror_ETD, label='ETD, alpha=' + str(best_ETD_step_size))
    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode\nAveraged over ' + str(len(Measures_TD)) + ' runs', rotation=0)
    plt.title('Best learning curve for TD and ETD control at ' + r'$lambda$ = ' + str(lam))
    plt.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    main()
