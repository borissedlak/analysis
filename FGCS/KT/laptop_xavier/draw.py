import os

import pandas as pd
from matplotlib import pyplot as plt

ROOT = os.path.dirname(__file__)
fig, ax = plt.subplots()

# kt_cpu = pd.read_csv(ROOT + '/laptop_xavier_kt_cpu.csv')
# kt_cpu['fact'] = kt_cpu['pv'] * kt_cpu['ra']
# scratch_cpu = pd.read_csv(ROOT + '/xavier_scratch_cpu.csv')
# scratch_cpu['fact'] = scratch_cpu['pv'] * scratch_cpu['ra']
# cpu_diff = (kt_cpu['fact'] - scratch_cpu['fact'])

kt_gpu = pd.read_csv(ROOT + '/laptop_xavier_kt_gpu.csv')
kt_gpu['fact'] = kt_gpu['pv'] * kt_gpu['ra']
scratch_gpu = pd.read_csv(ROOT + '/xavier_scratch_gpu.csv')
scratch_gpu['fact'] = scratch_gpu['pv'] * scratch_gpu['ra']
# gpu_diff = (kt_gpu['fact'] - scratch_gpu['fact'])

x = range(1, 101)
# plt.plot(x, cpu_diff.iloc[0:100], color='brown', label="Diff CPU")
plt.plot(x, kt_gpu['fact'].iloc[0:100], color='steelblue', label="Combined")
plt.plot(x, scratch_gpu['fact'].iloc[0:100], color='grey', label="Scratch")

kt_gpu['change_in_config'] = ((kt_gpu['pixel'] != kt_gpu['pixel'].shift(1))
                              & (kt_gpu['fps'] != kt_gpu['fps'].shift(1)))
scratch_gpu['change_in_config'] = ((scratch_gpu['pixel'] != scratch_gpu['pixel'].shift(1))
                                   & (scratch_gpu['fps'] != scratch_gpu['fps'].shift(1)))

first_label = True
for index, row in kt_gpu.iterrows():
    if row['change_in_config'] and index > 1:
        plt.scatter(index + 1, row['fact'], marker='o', color='blue', label="Config Change" if first_label else None)
        # plt.scatter(index + 1, row['ra'], marker='o', color='blue')
        first_label = False

first_label = True
for index, row in scratch_gpu.iterrows():
    if row['change_in_config'] and index > 1:
        plt.scatter(index + 1, row['fact'], marker='o', color='dimgray', label="Config Change" if first_label else None)
        # plt.scatter(index + 1, row['ra'], marker='o', color='blue')
        first_label = False

ax.set_xlim(0, 30)
fig.set_size_inches(4.5, 3.3)
ax.set_xlabel('ACI Cycle Iteration')
ax.set_ylabel('SLO Fulfillment (%)')
ax.legend()

plt.savefig("CPU_GPU_KT_Diff.eps", dpi=600, bbox_inches="tight", format="eps")
plt.show()

# ############################################################################################################
#
