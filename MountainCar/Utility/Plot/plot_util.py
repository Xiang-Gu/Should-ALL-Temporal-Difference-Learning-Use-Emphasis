import glob
import numpy as np
import re

INF = 999. # When performance measure is np.float('inf'), we replace it with this number so we can plot it to a graph
NUM_TRAINING_EPISODES = 50000 # Number of training episodes
COMMON_PATH = '/Users/rlai/Desktop/Research_UofA/MountainCar/'
# Collect data from cut_off_num to (cut_off_num + interval) for plot FP curve
CUT_OFF_NUM = 49949
INTERVAL = 50

# Extract lambda and step size from a path name
# e.g. ETD/Data/0.4/2-10/ --> ETD, 0.4, 0.0009765625 (2^-10), or
# /User/Home/Desktop/Research/Acrobot/TD/Data/0.9/2-11/ --> TD, 0.9, 0.00048828125 (2^-11)
def get_method_lam_alpha(directory):
    # Extract method
    method = 'TD' if directory.find('ETD') == -1 else 'ETD'

    # Extract lambda
    match = re.search(r'/[01].[0-9]/|/[01].[1-9][0-9]/', directory)
    if match:
        lam = float(match.group(0)[1:-1])
    else:
        assert False, 'Input directory does not contain lambda in proper format'

    # Extract alpha
    idx = directory.find('/2')
    if idx != -1:
        exp = ''
        for char in directory[idx+2:]: # Start with the exponents
            if char == '/':
                break
            exp += char
        alpha = pow(2., float(exp))
    else:
        alpha = 0.0 # Sometimes we just want to extract method and/or lambda information given a directory path like 'TD/Data/0.4/'

    return method, lam, alpha


# Used when we record multiple performance measures at each episode
# eachline: a list of strings where each string is one performance measure recorded at one episode
def get_each_line(eachline):
    result = float(eachline)
    if result == float('inf') or np.isnan(result): # If one file goes to infinity, just assume this parameter setting will make the algorithm diverge
        result = INF
    return result


# pad the last element (which is usually INF) to the end of
# items in lists whose size is less than length
# E.g. if lists = [[1,2,3,5], [1,5]] and length = 4,
# then the second item in lists will be padded to [1,5,5,5] and
# lists will thus be [[1,2,3,5],[1,5,5,5]]
def pad_incomplete_item(lists, length):
    for list in lists:
        if len(list) < length:
            list += [list[-1]] * (length - len(list))


# return a list of lists where each component list contains the performance measure of a trail run
# e.g. we run the same experiment for three times, and within each trial, we record the performance over 5 episodes
# Then it return something like [[9,6,4,2,1], [10,8,5,3,3,2], [10,7,5,3,1]]
# directory should be an absolute path for scalability
def get_all_files(directory):
    list_files = glob.glob(directory + '*')
    result = []

    for file_name in list_files:
        if not file_name.endswith('.err'):
            measure = [] # It stores the recorded performance of this file
            file = open(file_name, 'r')
            fline = file.readlines()
            # Read one line at a time and store it in measure
            for eachline in fline:
                eachline = get_each_line(eachline)
                measure.append(eachline)
            result.append(measure)

    pad_incomplete_item(result, NUM_TRAINING_EPISODES)
    return result


# Compute the mean and SEM of files in directory
# curve can be FP (Final Performance) or AUC (Area Under the Curve)
# directory should be one of step size directory (e.g. TD/Data/0.0/2-3/)
def compute_mean_and_SEM(directory, curve):
    measures = get_all_files(directory)
    means_each_file = []

    for measure in measures:
        if curve == 'FP':
            means_each_file.append(np.mean(measure[CUT_OFF_NUM : CUT_OFF_NUM + INTERVAL]))
        elif curve == 'AUC':
            means_each_file.append(np.mean(measure))
        else:
            assert False, 'input curve argument should be either FP or AUC'

    mean = np.mean(means_each_file)
    SEM = np.std(means_each_file)
    return mean, SEM

# Get the mean and SEM of all the different step sizes in directory
# E.g. directory = TD/Data/0.4/. Then this function will look at all the different
# step size directories and compute the mean and SEM of each step size directory.
# So the return will be something like [[0.25, 100, 2], [0.5, 110, 1.1], [0.125, 150, 1.0], ...]
def get_mean_and_SEM_all_stepSizes(directory, curve):
    step_size_directories = glob.glob(directory + '*/') # get a list of all directoies for different step sizes
    result = []

    for each_step_size_directory in step_size_directories:
        # Extract step size of current directory for TD method
        _, _, step_size = get_method_lam_alpha(each_step_size_directory)
        # Compute mean and SEM (of final performance) in this directory
        mean, SEM = compute_mean_and_SEM(each_step_size_directory, curve)
        # Add this data point [step_size, mean, SEM] to results
        result.append([step_size, mean, SEM])

    return result
