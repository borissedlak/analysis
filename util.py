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


def get_base_data_as_sample():
    ROOT = os.path.dirname(__file__)
    files = [
        # ROOT + f'/data/nano_cpu.csv',
        ROOT + f'/data/xavier_cpu_2_10.csv',
        ROOT + f'/data/xavier_cpu_4_15.csv',
        ROOT + f'/data/xavier_cpu_6_20.csv',
        ROOT + f'/data/xavier_gpu_2_10.csv',
        ROOT + f'/data/xavier_gpu_4_15.csv',
        ROOT + f'/data/xavier_gpu_6_20.csv',
    ]
    return pd.concat((pd.read_csv(f) for f in files))


def print_BN(bn: BayesianNetwork | pgmpy.base.DAG, root=None, try_visualization=False, vis_ls=None, save=False):
    if vis_ls is None:
        vis_ls = ["fdp"]
    else:
        vis_ls = vis_ls

    if try_visualization:
        vis_ls = ['neato', 'dot', 'twopi', 'fdp', 'sfdp', 'circo']

    for s in vis_ls:
        pos = graphviz_layout(bn, root=root, prog=s)
        nx.draw(
            bn, pos, with_labels=True, arrowsize=20, node_size=1500, alpha=0.8, font_weight="bold",
        )
        if save:
            plt.savefig(f"figures/{root}.png")
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
