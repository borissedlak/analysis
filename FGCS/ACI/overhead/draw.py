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

# r'[$\mathit{Xavier_{CPU}}$'
# r'[$\mathit{Xavier_{GPU}}$'

boxplot1 = ax.boxplot(df_bnl['cpu'], positions=[1], labels=['ACI'], patch_artist=True, widths=0.35)
boxplot2 = ax.boxplot(df_cpu_no_aci['cpu'], positions=[2], labels=['NO ACI'], patch_artist=True, widths=0.35)
boxplot3 = ax.boxplot(df_gpu_aci['cpu'], positions=[3], labels=['ACI'], patch_artist=True, widths=0.35)
boxplot4 = ax.boxplot(df_gpu_no_aci['cpu'], positions=[4], labels=['NO ACI'], patch_artist=True, widths=0.35)

boxplot1['boxes'][0].set_facecolor('mediumaquamarine')
boxplot2['boxes'][0].set_facecolor('mediumaquamarine')
boxplot3['boxes'][0].set_facecolor('steelblue')
boxplot4['boxes'][0].set_facecolor('steelblue')

ax.set_ylabel('CPU Utilization (%)')
# ax.set_ylabel('ACI Cycle Execution (ms)')
ax.set_title('')
ax.legend([boxplot1['boxes'][0], boxplot3['boxes'][0]], [r'$\mathit{Xavier_{CPU}}$', r'$\mathit{Xavier_{GPU}}$'])

fig.set_size_inches(4.5, 3.3)
# ax.set_ylim(85, 370)

# Show the plot
# plt.savefig("boxplot_ACI_overhead.eps", dpi=600, bbox_inches="tight", format="eps")  # default dpi is 100
# plt.show()

df_bnl = pd.read_csv(ROOT + '/training_time.csv')
df_bnl['duration'] = df_bnl['duration'] / 1000

for conf in ['pml', 'str']:

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
        if learn == conf:
            ax.bar(index, duration, color="cyan" if learn == "pml" else "brown",
               label=None)
        else:
            ax.bar(index, 0)

    # legend_labels = [("brown", "Structure Retrain"), ("cyan", "Parameter Retrain")]
    # legend_patches = [mlines.Line2D([], [], color=c, label=l, linewidth=8) for (c, l) in legend_labels]
    #
    # # Add the custom legend to the plot
    # ax.legend(legend_patches, ["Structure Retrain", "Parameter Retrain"], loc='upper left')

    # plt.yticks([0, 0.25, 0.35])
    # plt.xticks([0, 10, 20, 30, 40, 47])
    if conf == 'pml':
        fig.set_size_inches(5, 1.2)
        plt.yticks([0, 0.15, 0.3])
        ax.set_ylabel('PARL time (s)')
        ax.set_xlabel('ACI Cycle Iteration')
    else:
        fig.set_size_inches(5, 2.5)
        ax.set_ylabel('STRL time (s)')
        plt.xticks([])

    ax.set_xlim(-1, 100)

    # Show the plot
    plt.savefig(f"barplot_cycle_length_{conf}.eps", dpi=600, bbox_inches="tight", format="eps")  # default dpi is 100
    plt.show()
