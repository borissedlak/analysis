# Compare the SLO fulfillment when all on one
# Against when all on optimal
# Against when all on random
# Against equally distributed
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ROOT = os.path.dirname(__file__)
fig, ax = plt.subplots()
bar_width = 0.14
bar_gap = 0.0

category_color_map = {r'$\mathit{Laptop}$': 'firebrick', r'$\mathit{Nano}$': 'chocolate',
                      r'$\mathit{Xavier_{CPU}}$': 'mediumaquamarine',
                      r'$\mathit{Xavier_{GPU}}$': 'steelblue', r'$\mathit{Orin}$': 'dimgray'}
index_map = {r'$\mathit{Laptop}$': 0, r'$\mathit{Nano}$': 1,
             r'$\mathit{Xavier_{CPU}}$': 2,
             r'$\mathit{Xavier_{GPU}}$': 3, r'$\mathit{Orin}$': 4}

first_bar = True

for file, label, i in [(ROOT + '/single.csv', 'Single', 0), (ROOT + '/inferred.csv', 'Infer', 1),
                       (ROOT + '/random.csv', 'Rand', 2), (ROOT + '/equal.csv', 'Equal', 3)]:

    df = pd.read_csv(file)
    df['fact'] = df['pv'] * df['ra']
    df['GPU'] = df['GPU'].astype(str)
    df['id'] = df['device'] + df['GPU']

    result = df.groupby('id')['fact'].mean()
    result_dict = result.to_dict()

    result_dict[r'$\mathit{Laptop}$'] = result_dict.get('Laptop0', 0)
    result_dict[r'$\mathit{Xavier_{CPU}}$'] = result_dict.get('Xavier0', 0)
    result_dict[r'$\mathit{Xavier_{GPU}}$'] = result_dict.get('Xavier1', 0)
    result_dict[r'$\mathit{Orin}$'] = result_dict.get('Orin1', 0)
    result_dict[r'$\mathit{Nano}$'] = result_dict.get('Nano0', 0)

    del result_dict['Laptop0']
    del result_dict['Xavier0']
    del result_dict['Xavier1']
    del result_dict['Orin1']
    del result_dict['Nano0']

    categories = list(result_dict.keys())
    values = list(result_dict.values())

    # The issue is here, that it arranges the values every time from 0-4, it should
    # do so from
    x = [i - 0.275]

    for category, value in zip(categories, values):
        col = category_color_map[category]
        lab = category if first_bar else None
        plt.bar(x, value, label=lab, color=col, width=bar_width)  # Use x for the current category
        x = [pos + bar_width + bar_gap for pos in x]  # Move x position for the next category

    first_bar = False

plt.ylabel('Overall SLO fulfillment')
plt.legend(loc='lower left')
ax.set_ylim(0, 1)
fig.set_size_inches(5.5, 2.1)
plt.xticks(np.arange(4), ['Single', 'Infer', 'Rand', 'Equal'])

# Show the plot
plt.savefig("overall_slo_performance.png", dpi=600, bbox_inches="tight")  # default dpi is 100
plt.show()

fig, ax = plt.subplots()
first_bar = True

for file, label, j, dist in [(ROOT + '/single.csv', 'Single', 0, [5, 1, 1, 1, 1, 1]),
                          (ROOT + '/inferred.csv', 'Infer', 1, [25, 9, 1, 5, 9, 1]),
                          (ROOT + '/random.csv', 'Rand', 2, [25, 4, 5, 8, 4, 3]),
                          (ROOT + '/equal.csv', 'Equal', 3, [25, 5, 5, 5, 5, 5])]:

    df = pd.read_csv(file)
    df['fact'] = df['pv'] * df['ra']
    df['GPU'] = df['GPU'].astype(str)
    df['id'] = df['device'] + df['GPU']

    result = df.groupby('id')['fact'].mean()
    result_dict = result.to_dict()

    result_dict[r'$\mathit{Laptop}$'] = result_dict['Laptop0']
    result_dict[r'$\mathit{Xavier_{CPU}}$'] = result_dict['Xavier0']
    result_dict[r'$\mathit{Xavier_{GPU}}$'] = result_dict['Xavier1']
    result_dict[r'$\mathit{Orin}$'] = result_dict['Orin1']
    result_dict[r'$\mathit{Nano}$'] = result_dict['Nano0']
    del result_dict['Laptop0']
    del result_dict['Xavier0']
    del result_dict['Xavier1']
    del result_dict['Orin1']
    del result_dict['Nano0']

    categories = list(result_dict.keys())
    values = list(result_dict.values())

    bottom_a = 0
    bottom_b = 0
    i = 1

    x = j - 0.145


    for category, v in zip(categories, values):
        value = v / 5
        col = category_color_map[category]
        lab = category if first_bar else None
        plt.bar(x, value, bottom=bottom_a, label=lab, color=col, width=0.25)
        bottom_a += value

        value = v * (dist[i] / dist[0])
        col = category_color_map[category]
        plt.bar(x + 0.3, value, bottom=bottom_b, label=None, color=col, width=0.25)
        bottom_b += value
        i += 1

    first_bar = False

plt.ylabel('Average SLO fulfillment')
# plt.legend(loc='lower left')
ax.set_ylim(0, 1)
fig.set_size_inches(5.5, 2.1)
plt.xticks(np.arange(4), ['Single', 'Infer', 'Rand', 'Equal'])

# Show the plot
plt.savefig("overall_slo_performance_avg.png", dpi=600, bbox_inches="tight")  # default dpi is 100
plt.show()
