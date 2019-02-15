from matplotlib import pyplot as plt
import argparse
from plot_util import COMMON_PATH, get_mean_and_SEM_all_stepSizes

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lam', default=0.0, type=float,
                    help='compare both methods with this lambda')
parser.add_argument('--curve', default='FP', type=str,
                    help='method to plot (either FP or AUC)')
parser.add_argument('--display', default='original', type=str,
                    help='Axies display option (original or shift)')

args = parser.parse_args()
lam = args.lam
curve = args.curve
curve_verbose = 'Final Performance Curve' if curve == 'FP' else 'Area Under the Curve'
display = args.display
TD_path = COMMON_PATH + 'TD/Data/' + str(lam) + '/'
ETD_path = COMMON_PATH + 'ETD/Data/' + str(lam) + '/'

def main():
    result_TD = get_mean_and_SEM_all_stepSizes(TD_path, curve)
    result_ETD = get_mean_and_SEM_all_stepSizes(ETD_path, curve)

    # Sort the results of both method according to the step size
    result_TD.sort(key=lambda x : x[0])
    result_ETD.sort(key=lambda x : x[0])

    alphas_TD = [each_point[0] for each_point in result_TD]
    averages_TD = [each_point[1] for each_point in result_TD]
    errors_TD = [each_point[2] for each_point in result_TD]

    alphas_ETD = [each_point[0] for each_point in result_ETD]
    averages_ETD = [each_point[1] for each_point in result_ETD]
    errors_ETD = [each_point[2] for each_point in result_ETD]

    # Plot results according to display option
    if display == 'original':
        # Plot with the same scale of x-axis
        plt.title('MountainCar Control \n' + curve_verbose +'\n' + r'$\lambda$' + ' = ' + str(lam), fontsize=15)
        plt.xscale('log')
        plt.xlabel(r'$\alpha$', fontsize=20)
        plt.ylabel('Steps per episode \n Averaged over 30 runs', rotation=0)
        plt.errorbar(alphas_TD, averages_TD, yerr=errors_TD, fmt='-o', label='TD')
        plt.errorbar(alphas_ETD, averages_ETD, yerr=errors_ETD, fmt='--o', label='ETD')
        plt.legend(loc='best', fontsize=15)
        plt.show()
    elif display == 'shift':
        # Plot on differnt scale of x-axis
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xscale('log')
        ax1.set_xlabel(r'$\alpha$', fontsize=20)  # we already handled the x-label with ax2
        ax1.set_ylabel('Steps per episode \n Averaged over 30 runs', rotation=0)
        ax1.errorbar(alphas_ETD, averages_ETD, yerr=errors_ETD, fmt='--o', label='ETD', color=color)
        ax1.tick_params(axis='x', labelcolor=color)

        ax2 = ax1.twiny()  # instantiate a second axes that shares the same y-axis

        color = 'tab:blue'
        ax2.set_xscale('log')
        ax2.errorbar(alphas_TD, averages_TD, yerr=errors_TD, fmt='-o', label='TD', color=color)
        ax2.tick_params(axis='x', labelcolor=color)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0, fontsize=15)
        plt.title('MountainCar Control \n' + curve_verbose +'\n' + r'$\lambda$' + ' = ' + str(lam), fontsize=15)
        plt.show()

if __name__ == '__main__':
    main()
