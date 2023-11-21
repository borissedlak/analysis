import os

import pandas as pd
from matplotlib import pyplot as plt

ROOT = os.path.dirname(__file__)
fig, ax = plt.subplots()

surprise_all_devices = pd.read_csv(ROOT + '/surprise_per_device_model.csv')
# surprise_all_devices['surprise'] = surprise_all_devices['surprise'] / 45

surprise_nano = surprise_all_devices[surprise_all_devices['model'] == 'Nano'][:31]
surprise_orin = surprise_all_devices[surprise_all_devices['model'] == 'Orin'][:31]
surprise_laptop = surprise_all_devices[surprise_all_devices['model'] == 'Laptop'][:31]
surprise_xavier_cpu = surprise_all_devices[surprise_all_devices['model'] == 'Xavier CPU'][:31]
surprise_xavier_gpu = surprise_all_devices[surprise_all_devices['model'] == 'Xavier GPU'][:31]

surprise_nano = surprise_nano.reset_index(drop=True)
surprise_orin = surprise_orin.reset_index(drop=True)
surprise_laptop = surprise_laptop.reset_index(drop=True)
surprise_xavier_cpu = surprise_xavier_cpu.reset_index(drop=True)
surprise_xavier_gpu = surprise_xavier_gpu.reset_index(drop=True)

x = range(0, 31)
plt.plot(x, surprise_nano['surprise'], color='chocolate', label=r'Model of $\mathit{Nano}$')
plt.plot(x, surprise_orin['surprise'], color='dimgray', label=r'Model of $\mathit{Orin}$')
plt.plot(x, surprise_laptop['surprise'], color='firebrick', label=r'Model of $\mathit{Laptop}$')
plt.plot(x, surprise_xavier_cpu['surprise'], color='mediumaquamarine', label=r'Model $\mathit{Xavier_{CPU}}$')
plt.plot(x, surprise_xavier_gpu['surprise'], color='steelblue', label=r'Combined Model')

ax.set_xlim(0, 25)
# ax.set_xticks([])
ax.set_ylim(25, 125)
ax.set_ylim(25, 125)
fig.set_size_inches(5, 3.3)
ax.set_ylabel('BIC Surprise')
ax.legend()

plt.savefig("surprise_per_model.eps", dpi=600, bbox_inches="tight", format="eps")
plt.show()

# #########################################################################################################

fig, ax = plt.subplots()

first_label = True
for i, (df, c) in enumerate(
        [(surprise_xavier_gpu, 'steelblue'), (surprise_laptop, 'firebrick'), (surprise_xavier_cpu, 'mediumaquamarine'),
         (surprise_orin, 'dimgray'), (surprise_nano, 'chocolate')]):
    for index, row in df.iterrows():
        if row['retrain']:
            plt.scatter(index, i + 100.75, marker='v', color=c,
                        label="Config Change" if first_label else None)
            first_label = False

ax.set_xlim(0, 25)
ax.set_ylim(100, 105.75)
ax.set_yticks([100])
ax.set_xlabel('ACI Cycle Iteration')
ax.set_ylabel('CPT Retrain')
fig.set_size_inches(5, 1.2)
plt.savefig("surprise_per_model_train.eps", dpi=600, bbox_inches="tight", format="eps")
plt.show()

# #########################################################################################################
