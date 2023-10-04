import os

import pandas as pd
from matplotlib import pyplot as plt

ROOT = os.path.dirname(__file__)
df_initial = pd.read_csv(ROOT + '/initial_training.csv')
df_slo_change = pd.read_csv(ROOT + '/slo_change.csv')
df_slo_change_s = pd.read_csv(ROOT + '/slo_change_surprise.csv')
df_stream_change = pd.read_csv(ROOT + '/stream_change.csv')
df_video_change = pd.read_csv(ROOT + '/video_change.csv')

# fig, ax = plt.subplots()
#
# x = range(1, len(df_initial) + 1)
# plt.plot(x, df_initial['pv'], color='green', label="PV SLOs")
# plt.plot(x, df_initial['ra'], color='red', label="RA SLOs")
# ax.set_xlabel('ACI Cycle Iteration')
# ax.set_ylabel('SLO Fulfillment Rate')
#
# df_initial['change_in_config'] = ((df_initial['pixel'] != df_initial['pixel'].shift(1))
#                                   & (df_initial['fps'] != df_initial['fps'].shift(1)))
#
# first_label = True
# for index, row in df_initial.iterrows():
#     if row['change_in_config']:
#         plt.scatter(index + 1, row['pv'], marker='o', color='blue', label="Conf Change" if first_label else None)
#         plt.scatter(index + 1, row['ra'], marker='o', color='blue')
#         first_label = False
#
# ax.set_xticks(range(0, 50, 4))
# fig.set_size_inches(4.5, 3.3)
# ax.set_xlim(0, 20)
# ax.set_ylim(0.45, 1.03)
# ax.legend()
#
# # Show the plot
# plt.savefig("cycle_iteration.png", dpi=600, bbox_inches="tight")  # default dpi is 100
# plt.show()
#
# ############################################################################################################
#
# fig, ax = plt.subplots()
#
# x = range(1, len(df_slo_change) + 1)
# plt.plot(x, df_slo_change['pv'], color='green', label="PV SLOs")
# plt.plot(x, df_slo_change['ra'], color='red', label="RA SLOs")
# # ax.set_xlabel('ACI Cycle Iteration')
# ax.set_ylabel('SLO Fulfillment Rate')
#
# df_slo_change['change_in_config'] = ((df_slo_change['pixel'] != df_slo_change['pixel'].shift(1))
#                                      & (df_slo_change['fps'] != df_slo_change['fps'].shift(1)))
#
# first_label = True
# for index, row in df_slo_change.iterrows():
#     if row['change_in_config'] and index > 5:
#         plt.scatter(index + 1, row['pv'], marker='o', color='blue', label="Conf Change" if first_label else None)
#         plt.scatter(index + 1, row['ra'], marker='o', color='blue')
#         first_label = False
#
# plt.scatter([3, 3], [df_slo_change.iloc[2]['pv'], df_slo_change.iloc[2]['ra']], marker='*', color='orange',
#             label="SLO Change")
#
# ax.set_xticks([])
# fig.set_size_inches(4.5, 2.1)
# ax.set_xlim(0, 40)
# ax.set_ylim(0.25, 1.03)
# ax.legend()
#
# # Show the plot
# plt.savefig("slo_change.png", dpi=600, bbox_inches="tight")  # default dpi is 100
# plt.show()
#
# ############################################################################################################
#
# fig, ax = plt.subplots()
#
# plt.bar(range(len(df_slo_change_s)), df_slo_change_s['surprise'])
# plt.xticks(range(len(df_slo_change_s)), df_slo_change_s.index)
#
# plt.scatter([3, 4, 5], [df_slo_change_s.iloc[3:6]['surprise']], marker='^', color='brown', label="Structure Retrain")
# plt.scatter([6, 7, 8], [df_slo_change_s.iloc[6:9]['surprise']], marker='v', color='cyan', label="Parameter Retrain")
#
# ax.set_xticks(range(0, 50, 4))
# fig.set_size_inches(4.5, 2.1)
# ax.set_xlim(0, 40)
# ax.set_ylim(0, 105)
# ax.set_xlabel('ACI Cycle Iteration')
# ax.set_ylabel('BIC Surprise')
# ax.legend()
#
# plt.savefig("slo_change_surprise.png", dpi=600, bbox_inches="tight")
# plt.show()

############################################################################################################

for df, name in [(df_stream_change, "stream_change.png"), (df_video_change, "video_change.png")]:

    fig, ax = plt.subplots()

    x = range(1, len(df) + 1)
    plt.plot(x, df['pv'], color='green', label="PV SLOs")
    plt.plot(x, df['ra'], color='red', label="RA SLOs")

    df['change_in_config'] = ((df['pixel'] != df['pixel'].shift(1))
                              & (df['fps'] != df['fps'].shift(1)))

    first_label = True
    for index, row in df.iterrows():
        if row['change_in_config'] and index > 1:
            plt.scatter(index + 1, row['pv'], marker='o', color='blue', label="Conf Change" if first_label else None)
            plt.scatter(index + 1, row['ra'], marker='o', color='blue')
            first_label = False

    plt.scatter([3, 3], [df.iloc[2]['pv'], df.iloc[2]['ra']], marker='*', color='orange',
                label="SLO Change")

    ax.set_xlim(0, 20)
    ax.set_ylim(-0.03, 1.05)
    fig.set_size_inches(3.5, 3.3)
    ax.set_xlabel('ACI Cycle Iteration')
    ax.set_ylabel('SLO Fulfillment Rate')
    ax.legend()

    plt.savefig(name, dpi=600, bbox_inches="tight")
    plt.show()
