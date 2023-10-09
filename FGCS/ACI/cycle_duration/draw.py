import os

import pandas as pd
from matplotlib import pyplot as plt

ROOT = os.path.dirname(__file__)
df_full = pd.read_csv(ROOT + '/full.csv')
df_mb_full = pd.read_csv(ROOT + '/mb_full.csv')
df_mb_half = pd.read_csv(ROOT + '/mb_half.csv')
df_mb_one = pd.read_csv(ROOT + '/mb_one.csv')

fig, ax = plt.subplots()

boxplot1 = ax.boxplot(df_full, positions=[1], labels=['No MB'], patch_artist=True, widths=0.3)
boxplot2 = ax.boxplot(df_mb_full, positions=[2], labels=['MB [4 SLOs'], patch_artist=True, widths=0.3)
boxplot3 = ax.boxplot(df_mb_half, positions=[3], labels=['2 SLOs'], patch_artist=True, widths=0.3)
boxplot4 = ax.boxplot(df_mb_half, positions=[4], labels=['1 SLO]'], patch_artist=True, widths=0.3)

boxplot1['boxes'][0].set_facecolor('lightslategray')
boxplot2['boxes'][0].set_facecolor('steelblue')
boxplot3['boxes'][0].set_facecolor('steelblue')
boxplot4['boxes'][0].set_facecolor('steelblue')

# ax.set_xlabel(' ')
ax.set_ylabel('ACI Cycle Execution (ms)')
# ax.set_title('')
# ax.legend([boxplot1['boxes'][0], boxplot2['boxes'][0]], ['Thirty Values', 'Full Data'])

fig.set_size_inches(4.5, 3.3)
ax.set_ylim(85, 370)

# Show the plot
plt.savefig("boxplot_ACI_cycle.png", dpi=600, bbox_inches="tight")  # default dpi is 100
plt.show()
