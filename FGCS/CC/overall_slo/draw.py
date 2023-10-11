# Compare the SLO fulfillment when all on one
# Against when all on optimal
# Against when all on random
# Against equally distributed
import os

import pandas as pd
from matplotlib import pyplot as plt

ROOT = os.path.dirname(__file__)
fig, ax = plt.subplots()

for file, label in [(ROOT + '/single.csv', 'Single'), (ROOT + '/inferred.csv', 'Inferred'),
                    (ROOT + '/random.csv', 'Random'), (ROOT + '/equal.csv', 'Equal')]:

    df = pd.read_csv(file)
    df['fact'] = df['pv'] * df['ra']
    df['GPU'] = df['GPU'].astype(str)
    df['id'] = df['device'] + df['GPU']

    result = df.groupby('id')['fact'].mean()
    result_dict = result.to_dict()

    categories = list(result_dict.keys())
    values = list(result_dict.values())

    bottom = 0

    for category, value in zip(categories, values):
        plt.bar(label, value, bottom=bottom, label=category, width=0.4)
        bottom += value

plt.ylabel('Overll SLO fulfillment')
plt.legend()
ax.set_ylim(0, 5)
fig.set_size_inches(4.5, 3.3)

# Show the plot
plt.show()
