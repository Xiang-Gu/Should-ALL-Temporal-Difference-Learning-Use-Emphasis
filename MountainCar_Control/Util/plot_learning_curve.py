from matplotlib import pyplot as plt
import argparse
import numpy as np
import glob
import math

# Extract lambda and step size from a path name
# e.g. ETD/Data/0.4/2-10/ --> 0.4, 0.0009765625 (2^-10)
def get_lam_and_alpha(directory):
    method = ''
    lam = ''
    alpha = ''
    flag_lam = 0
    flag_alpha = False
    # Extract method (TD or ETD)
    for char in directory:
        if char == '/' or char == '\\':
            break
        method += char

    # Extract lam
    for char in reversed(directory):
        if (char == '/' or char == '\\'):
            flag_lam += 1
        elif flag_lam == 2:
            lam += char

    # Extract alpha
    for char in reversed(directory):
        if (char == '/' or char == '\\') and not flag_alpha:
            flag_alpha = True
        elif (char == '/' or char == '\\') and flag_alpha:
            break
        elif flag_alpha:
            alpha += char

    # Reverse the lam and alpha string
    lam = lam[::-1]
    alpha = alpha[::-1]

    # convert alpha from string form to numerical number (e.g. 2-10 --> 0.0009765625)
    base = float(alpha[0])
    exp = float(alpha[1:])


    return method, float(lam), pow(base, exp)


# Read data from a directory and plot the results
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--directory', default='TD/Data/0.0/2-8/', type=str,
                    help='directory that contains files in different runs')


    args = parser.parse_args()
    method, lam, alpha = get_lam_and_alpha(args.directory)

    list_files = glob.glob(args.directory + '*') # get a list of all files in onpolicy_results_directory
    num_trials = len(list_files) # Number of files (# of trials we ran) in the directory

    # Measures is a list of lists, where each element (each list)
    # is the Measures of a trial
    Measures = []

    for idx in range(num_trials):
        file_name = list_files[idx]
        if not file_name.endswith('.err'):
            Measure_current_trial = []
            file = open(file_name, 'r')
            fline = file.readlines()
            # Add each line (which is one float) to Measures
            for eachline in fline:
                Measure_current_trial.append(float(eachline))
            Measures.append(Measure_current_trial)

    # Plot the graph
    mean = np.mean(Measures, axis=0)
    plt.yscale('log')
    plt.plot(np.arange(mean.size), mean, label=r'$\lambda$' + '=' + str(lam) + ', alpha=' + str(alpha))
    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode\nAveraged over ' + str(len(Measures)) + ' runs', rotation=0)
    plt.title('learning curve for ' + method + '(' + str(lam) + ') control')
    plt.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    main()
