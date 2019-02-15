from matplotlib import pyplot as plt
import argparse
import numpy as np
from plot_util import get_method_lam_alpha, COMMON_PATH, get_all_files

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--directory', default='TD/Data/0.0/2-8/', type=str,
                    help='directory that contains files of different runs')

args = parser.parse_args()
directory = COMMON_PATH + args.directory # Turn directory into an absolute path
method, lam, alpha = get_method_lam_alpha(directory) # Extract method, lambda and step size from the directory path


# Read data from a directory and plot the results
def main():
    # measures is a list of lists, where each element (each list)
    # is the measures of a trial
    measures = get_all_files(directory)

    # Plot the graph
    mean = np.mean(measures, axis=0)
    plt.plot(np.arange(mean.size), mean, label=r'$\lambda$' + '=' + str(lam) + ', alpha=' + str(alpha))
    plt.xlabel('Episodes')
    plt.ylabel('Steps per Episode\nAveraged over ' + str(len(measures)) + ' runs', rotation=0)
    plt.title('Learning Curve for ' + method + ' Control on MountainCar')
    plt.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    main()
