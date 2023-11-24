import csv
import os

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'serif'

ROOT = os.path.dirname(__file__)
filename = ROOT + '/batch_size_history.csv'
max_length = 19

batch_size_data = []
with open(filename, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        batch_size_data.append([int(cell) for cell in row])

batch_size_data = [arr[:max_length] for arr in batch_size_data]
fig, ax = plt.subplots()
x_values = range(1, len(batch_size_data[0]) + 1)

legend_text = ["agent₁₂", "agent₃₀", "Increment (±1)"]

# Plot each array
for idx, arr in enumerate(batch_size_data):
    ax.plot(x_values, arr, label=f'{legend_text[idx]}')
    ax.scatter(x_values, arr, marker='o')

# Add labels, legend, and title
ax.set_xticks(range(1, max_length + 1, 2))
ax.set_yticks(range(12, 31, 3))
ax.set_xlabel('Active Inference Cycle')
ax.set_ylabel('Configured Batch Size')
# ax.set_title('History of batch sizes per ACI cycle')
ax.legend()

ax.set_aspect('auto', adjustable='box')
fig.set_size_inches(6, 4)

# plt.box(False)
plt.savefig(f"batch_sizes.eps", dpi=600, bbox_inches="tight", format="eps")  # default dpi is 100
plt.show()

data_interpolated = {
    13: 96.99248120300751, 14: 93.98496240601504, 15: 90.97744360902256,
    16: 87.96992481203007, 17: 84.96240601503759, 18: 81.95488721804512,
    25: 24.501449275362327,
    26: 19.591304347826086, 27: 14.843478260869565, 28: 9.722222222222221, 29: 4.861111111111111
}

data_experienced = {
    12: 100.0,
    19: 78.94736842105263, 20: 75.0, 21: 87.75510204081633, 22: 63.63636363636363,
    23: 65.21739130434783, 24: 29.166666666666668, 30: 0.0
}

fig, ax = plt.subplots()

tuples_i = [(key, 100 - value) for key, value in data_interpolated.items()]
x, y = zip(*tuples_i)
ax.scatter(x, y, marker='o', color="red", label="Interpolated")

tuples_e = [(key, 100 - value) for key, value in data_experienced.items()]
x, y = zip(*tuples_e)
ax.scatter(x, y, marker='o', color="blue", label="Experienced")

merged = (tuples_i + tuples_e)
merged.sort(key=lambda q: q[0])
x_values, y_values = zip(*merged)
baseline = -1  # Adjust this if needed
plt.fill_between(np.array(x_values), baseline, np.array(y_values), where=(np.array(y_values) > baseline), alpha=0.5,
                 label='Expected Risk')

fig.set_size_inches(4, 4)
ax.legend()
ax.set_xticks(range(12, 31, 3))
ax.set_xlabel('Possible Batch Size')
ax.set_ylabel('Assigned Risk')


plt.savefig(f"risk_assigned.eps", dpi=600, bbox_inches="tight", format="eps")  # default dpi is 100
plt.show()