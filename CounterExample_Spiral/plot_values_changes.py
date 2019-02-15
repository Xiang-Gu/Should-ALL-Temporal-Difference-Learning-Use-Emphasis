import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
TD_data = np.loadtxt('TD_results_values')
ETD_data = np.loadtxt('ETD_results_values')

TD_x = TD_data[:,0]
TD_y = TD_data[:,1]
TD_z = TD_data[:,2]

ETD_x = ETD_data[:,0]
ETD_y = ETD_data[:,1]
ETD_z = ETD_data[:,2]

# ax.plot(TD_x, TD_y, TD_z, label='TD')
# ax.plot(ETD_x, ETD_y, ETD_z, label='ETD')

ax.plot(TD_x, TD_y, TD_z, color='tab:blue')
ax.plot(ETD_x, ETD_y, ETD_z, color='tab:red')
# ax.set_title('Change of estimated values under onpolicy TD(0) and onpolicy ETD(0) learning')
ax.set_xlabel(r'$\hat{v}(s_1, w)$', fontsize=25, labelpad=15)
ax.set_ylabel(r'$\hat{v}(s_2, w)$', fontsize=25, labelpad=15)
ax.set_zlabel(r'$\hat{v}(s_3, w)$', fontsize=25, labelpad=15, rotation=0)
ax.tick_params(labelsize=23, which='major', axis='both')
# ax.legend()

plt.show()
