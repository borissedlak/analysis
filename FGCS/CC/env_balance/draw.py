import os

import pandas as pd
from matplotlib import pyplot as plt

ROOT = os.path.dirname(__file__)
fig, ax = plt.subplots()

data_laptop = pd.read_csv(ROOT + '/Laptop_l2.csv')
data_orin = pd.read_csv(ROOT + '/Orin.csv')

data_laptop['factor'] = data_laptop['pv'] * data_laptop['ra']
data_orin['factor'] = data_orin['pv'] * data_orin['ra']

x = range(0, 51)
plt.plot(x, data_laptop['factor'][:51], color='firebrick', label=r'$\mathit{Laptop}$')
plt.plot(x, data_orin['factor'][:51], color='dimgray', label=r'$\mathit{Orin}$')


plt.scatter(10, data_laptop['factor'][10], marker='s', color='orange', label="Net. Issue")
plt.scatter(30, data_laptop['factor'][30], marker='s', color='blue', label="Rebalance")

ax.set_xlim(0, 50)
# ax.set_xticks([])
# ax.set_ylim(25, 125)
fig.set_size_inches(5, 3.3)
ax.set_ylabel('SLO fulfillment rate')
ax.set_xlabel('ACI Cycle Iteration')
ax.legend()

plt.savefig("rebalancing.eps", bbox_inches="tight", format="eps")
plt.show()
