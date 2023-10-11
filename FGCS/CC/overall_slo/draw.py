# Compare the SLO fulfillment when all on one
# Against when all on optimal
# Against when all on random
# Against equally distributed
import os

import pandas as pd
from matplotlib import pyplot as plt

ROOT = os.path.dirname(__file__)
fig, ax = plt.subplots()

category_color_map = {r'$\mathit{Laptop}$': 'firebrick', r'$\mathit{Nano}$': 'chocolate',
                      r'$\mathit{Xavier_{CPU}}$': 'mediumaquamarine',
                      r'$\mathit{Xavier_{GPU}}$': 'steelblue', r'$\mathit{Orin}$': 'dimgray'}

first_bar = True

for file, label in [(ROOT + '/single.csv', 'Single'), (ROOT + '/inferred.csv', 'Infer'),
                    (ROOT + '/random.csv', 'Rand'), (ROOT + '/equal.csv', 'Equal')]:

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

    bottom = 0

    for category, value in zip(categories, values):
        col = category_color_map[category]
        lab = category if first_bar else None
        plt.bar(label, value, bottom=bottom, label=lab, color=col, width=0.5)
        bottom += value

    first_bar = False

# bottom = 0
# for category, value in [('Laptop0', 0.63), ('Nano0', 0.73), ('Orin1', 0.62),
#                         ('Xavier0', 0.62), ('Xavier1', 0.61)]:
#     col = category_color_map[category]
#     lab = category if first_bar else None
#     plt.bar('Expectation', value, bottom=bottom, label=lab, color=col, width=0.4)
#     bottom += value

plt.ylabel('Overall SLO fulfillment')
plt.legend()
ax.set_ylim(0, 4.2)
fig.set_size_inches(4.5, 3.3)

# Show the plot
plt.savefig("overall_slo_performance.png", dpi=600, bbox_inches="tight")  # default dpi is 100
plt.show()
