import os

import pandas as pd
from matplotlib import lines as mlines
from matplotlib import pyplot as plt

ROOT = os.path.dirname(__file__)
df_bnl = pd.read_csv(ROOT + '/cpu_aci.csv')
df_cpu_no_aci = pd.read_csv(ROOT + '/cpu_no_aci.csv')
df_gpu_aci = pd.read_csv(ROOT + '/gpu_aci.csv')
df_gpu_no_aci = pd.read_csv(ROOT + '/gpu_no_aci.csv')

df_bnl = df_bnl[df_bnl['cpu'] != 0]
df_cpu_no_aci = df_cpu_no_aci[df_cpu_no_aci['cpu'] != 0]
df_gpu_aci = df_gpu_aci[df_gpu_aci['cpu'] != 0]
df_gpu_no_aci = df_gpu_no_aci[df_gpu_no_aci['cpu'] != 0]

fig, ax = plt.subplots()

boxplot1 = ax.boxplot(df_bnl['cpu'], positions=[1], labels=['CPU ACI'], patch_artist=True, widths=0.35)
boxplot2 = ax.boxplot(df_cpu_no_aci['cpu'], positions=[2], labels=['CPU NO'], patch_artist=True, widths=0.35)
boxplot3 = ax.boxplot(df_gpu_aci['cpu'], positions=[3], labels=['GPU ACI'], patch_artist=True, widths=0.35)
boxplot4 = ax.boxplot(df_gpu_no_aci['cpu'], positions=[4], labels=['GPU NO'], patch_artist=True, widths=0.35)

boxplot1['boxes'][0].set_facecolor('brown')
boxplot2['boxes'][0].set_facecolor('brown')
boxplot3['boxes'][0].set_facecolor('purple')
boxplot4['boxes'][0].set_facecolor('purple')

ax.set_ylabel('CPU Utilization (%)')
# ax.set_ylabel('ACI Cycle Execution (ms)')
ax.set_title('')
# ax.legend([boxplot1['boxes'][0], boxplot2['boxes'][0]], ['Thirty Values', 'Full Data'])

fig.set_size_inches(4.5, 3.3)
# ax.set_ylim(85, 370)

# Show the plot
plt.savefig("boxplot_ACI_overhead.png", dpi=600, bbox_inches="tight")  # default dpi is 100
plt.show()

df_bnl = pd.read_csv(ROOT + '/training_time.csv')
df_bnl['duration'] = df_bnl['duration'] / 1000

first_row = True
fig, ax = plt.subplots()
for index, row in df_bnl.iterrows():
    duration = row['duration']
    learn = row['type']
    if first_row:
        # label = "Structure Retrain" if learn == "pml" else "Parameter Retrain"
        first_row = False
    else:
        label = None
    ax.bar(index, duration, color="cyan" if learn == "pml" else "brown",
           label=None)

legend_labels = [("brown", "Structure Retrain"), ("cyan", "Parameter Retrain")]
legend_patches = [mlines.Line2D([], [], color=c, label=l, linewidth=8) for (c, l) in legend_labels]

# Add the custom legend to the plot
ax.legend(legend_patches, ["Structure Retrain", "Parameter Retrain"], loc='upper left')

plt.yticks([0.25, 5, 10, 15, 20])
plt.xticks([0, 10, 20, 30, 40, 47])
fig.set_size_inches(5, 3.3)
ax.set_xlim(-1, 59)
ax.set_ylabel('BNL Training Time (s)')
ax.set_xlabel('ACI Cycle Iteration')

# Show the plot
plt.savefig("barplot_cycle_length.png", dpi=600, bbox_inches="tight")  # default dpi is 100
plt.show()
