from matplotlib import pyplot as plt
import argparse
import numpy as np
import glob


'''
Go to directory Data/ and plot data of files stored in that directory
Make sure you store all the files in a directory called Data/
'''
TD_alpha = pow(2, -14)
ETD_alpha = pow(2, -17)
lam = 0.0
gamma = 0.9

def main():
    TD_list_files = glob.glob('TD/Data/*') # get a list of all files in onpolicy_results_directory
    ETD_list_files = glob.glob('ETD/Data/*') # get a list of all files in onpolicy_results_directory

    # Check point 1: whether we get the correct file names
    print('list of files = ' + str(TD_list_files))
    # Check point 2: whether we get the correct file names
    print('list of files = ' + str(ETD_list_files))

    TD_measures = []
    ETD_measures = []

    num_trials = len(TD_list_files)
    for idx in range(num_trials):
        file_name = TD_list_files[idx]
        if not file_name.endswith('.err'):
            measure = []
            file = open(file_name, 'r')
            fline = file.readlines()
            # Add each line (which is one float) to TD_measures
            for eachline in fline:
                measure.append(float(eachline))
            TD_measures.append(measure)

    num_trials = len(ETD_list_files)
    for idx in range(num_trials):
        file_name = ETD_list_files[idx]
        if not file_name.endswith('.err'):
            measure = []
            file = open(file_name, 'r')
            fline = file.readlines()
            # Add each line (which is one float) to TD_measures
            for eachline in fline:
                measure.append(float(eachline))
            ETD_measures.append(measure)



    # Plot the graph
    TD_mean = np.mean(TD_measures, axis=0)
    ETD_mean = np.mean(ETD_measures, axis=0)
    fig, ax = plt.subplots(nrows=1,ncols=1)
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000], [0, '1K', '2K', '3K', '4K', '5K', '6K'])
    ax.plot(np.arange(TD_mean.size), TD_mean, color='tab:blue')
    ax.plot(np.arange(ETD_mean.size), ETD_mean, color='tab:red')

    # plt.plot(np.arange(TD_mean.size), TD_mean, label=r'TD: $\alpha$ = ' + str(TD_alpha) + r', $\lambda$ = 0' +  r', $\gamma$ = ' + str(gamma))
    # plt.plot(np.arange(ETD_mean.size), ETD_mean, label=r'ETD: $\alpha$ = ' + str(ETD_alpha) + r', $\lambda$ = 0' + r', $\gamma$ = ' + str(gamma))
    ax.set_xlabel('Episodes', fontsize=25)
    ax.set_ylabel('Weight', rotation=0, fontsize=25, labelpad=10)
    ax.set_xlim(left=0, right=6000)
    ax.tick_params(labelsize=23, which='major', axis='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('/Users/rlai/Desktop/test3.pdf', format='pdf', dpi=300, bbox_inches='tight')
    # plt.title('Change of approximator parameter in Tistisklis and Van Roy\'s counterexample')
    # plt.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    main()
