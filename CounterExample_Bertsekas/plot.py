import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 51, 1)

# case 1
# y_true = np.arange(0, 51, 1)
# y_true[50] = 0
#
# y_TD1 = 0.94176 * x
# y_TD0 = -0.9607 * x
# y_ETD0 = 0.886879 * x

# case 2
y_true = np.ones(51)
y_true[0] = 0

y_TD1 = 0.0297 * x
y_TD0 = 0.0007843 * x
y_ETD0 = 0.002262 * x


# red dashes, blue squares and green triangles
fig, ax = plt.subplots(nrows=1,ncols=1)
ax.plot(x, y_true, label='True Value', color='tab:brown')

ax.plot(x, y_TD1, '--', label='TD(1)', color='tab:cyan')
ax.plot(x, y_TD1, '--', label='ETD(1)', color='tab:pink')

ax.plot(x, y_TD0, '--',  label='TD(0)', color='tab:blue')
ax.plot(x, y_ETD0, '--', label='ETD(0)', color='tab:red')

ax.set_xlabel('State', fontsize=25)
ax.set_ylabel('State \nValue', fontsize=25, rotation=0, labelpad=30)
ax.set_xlim(left=0, right=50)
ax.tick_params(labelsize=23, which='major', axis='both')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig('/Users/rlai/Desktop/test4.pdf', format='pdf', dpi=300, bbox_inches='tight')
# plt.title('Bertsekas\'s counterexample \n Case 2')
# plt.legend(loc=0)
plt.show()
