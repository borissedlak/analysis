import csv
import os

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

legend_text = ["Start Low  (12)", "Start High (30)", "Increment (±1)"]

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
plt.savefig(f"batch_sizes.png", dpi=400, bbox_inches="tight")  # default dpi is 100
plt.show()