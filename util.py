import copy
import os

import networkx
import networkx as nx
import numpy as np
import pandas as pd
import pgmpy
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from pgmpy.base import DAG
from pgmpy.models import BayesianNetwork
from sklearn.metrics import f1_score

regular = '#a1b2ff'  # blue
special = '#c46262'  # red


def get_prepared_base_samples(file=None, train=True):
    ROOT = os.path.dirname(__file__)
    if file is None:
        files = [
            ROOT + f'/data/laptop_cpu.csv',
            ROOT + f'/data/nano_cpu.csv',
            ROOT + f'/data/xavier_cpu_2_10.csv',
            ROOT + f'/data/xavier_cpu_4_15.csv',
            ROOT + f'/data/xavier_cpu_6_20.csv',
            ROOT + f'/data/xavier_gpu_2_10.csv',
            ROOT + f'/data/xavier_gpu_4_15.csv',
            ROOT + f'/data/xavier_gpu_6_20.csv',
        ]
    else:
        files = [ROOT + file]
    samples = pd.concat((pd.read_csv(f) for f in files))

    # Sanity check
    # print(samples.isna().any())
    samples = samples[samples['consumption'].notna()]
    samples.rename(columns={'execution_time': 'delay', 'success': 'transformed'}, inplace=True)

    samples['in_time'] = samples['delay'] <= (1000 / samples['fps'])

    samples['distance'] = samples['distance'].astype(int)

    samples['CPU'] = pd.cut(samples['cpu_utilization'], bins=[0, 50, 70, 90, 100],
                            labels=['Low', 'Mid', 'High', 'Very High'], include_lowest=True)
    samples['memory'] = pd.cut(samples['memory_usage'], bins=[0, 50, 70, 90, 100],
                               labels=['Low', 'Mid', 'High', 'Very High'], include_lowest=True)

    # samples['delay'] = pd.cut(samples['delay'], bins=7)
    # samples['distance'] = pd.cut(samples['distance'], bins=7)

    samples['bitrate'] = samples['fps'] * samples['pixel']
    samples['bitrate'] = samples['bitrate'].astype(str)
    samples['GPU'] = samples['GPU'].astype(bool)

    samples['distance_SLO_hard'] = pd.cut(samples['distance'], bins=[0, 35, max(samples['distance'])],
                                          labels=[True, False], include_lowest=True)
    samples['distance_SLO_easy'] = pd.cut(samples['distance'], bins=[0, 57, max(samples['distance'])],
                                          labels=[True, False], include_lowest=True)
    samples['time_slo'] = samples['delay'] <= (1000 / samples['fps'])

    del samples['timestamp']
    del samples['cpu_utilization']
    del samples['memory_usage']
    del samples['device_type']

    percentage_samples = 0.80 if train else 0.20
    return samples.sample(n=int(len(samples) * percentage_samples), random_state=35)


def print_BN(bn: BayesianNetwork | pgmpy.base.DAG, root=None, try_visualization=False, vis_ls=None, save=False,
             name=None, show=True, color_map=None):
    if vis_ls is None:
        vis_ls = ["fdp"]
    else:
        vis_ls = vis_ls

    if name is None:
        name = root

    if try_visualization:
        vis_ls = ['neato', 'dot', 'twopi', 'fdp', 'sfdp', 'circo']

    plt.figure(figsize=(6, 4.5))

    for s in vis_ls:
        pos = graphviz_layout(bn, root=root, prog=s)
        nx.draw(
            bn, pos, with_labels=True, arrowsize=20, node_size=1500,  # alpha=1.0, font_weight="bold",
            node_color=color_map
        )
        if save:
            plt.box(False)
            plt.savefig(f"figures/{name}.png", dpi=400, bbox_inches="tight")  # default dpi is 100
        if show:
            plt.show()


# Funtion to evaluate the learned model structures.
def get_f1_score(em, true_model):
    nodes = em.nodes()
    est_adj = nx.to_numpy_array(
        em.to_undirected(), nodelist=nodes, weight=None
    )
    true_adj = nx.to_numpy_array(
        true_model.to_undirected(), nodelist=nodes, weight=None
    )

    f1 = f1_score(np.ravel(true_adj), np.ravel(est_adj))
    print("F1-score for the model skeleton: ", f1)


def get_mb_as_bn(model: DAG | BayesianNetwork, center: str):
    mb_list = model.get_markov_blanket(center)
    mb = copy.deepcopy(model)

    for n in model.nodes:
        if n not in center.join(mb_list):
            mb.remove_node(n)

    return mb


def fix_edge_between_u_v(dag: DAG, u, v):
    try:
        dag.remove_edge(u, v)
    except networkx.exception.NetworkXError:
        dag.remove_edge(v, u)

    dag.add_edge(u, v)
    return dag
