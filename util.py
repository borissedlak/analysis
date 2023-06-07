import copy

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from pgmpy.models import BayesianNetwork
from sklearn.metrics import f1_score


def print_BN(bn: BayesianNetwork, root=None, try_visualization=False, vis_ls=None):

    if vis_ls is None:
        vis_ls = ["fdp"]
    else:
        vis_ls = vis_ls

    if try_visualization:
        vis_ls = ['neato', 'dot', 'twopi', 'fdp', 'sfdp', 'circo']

    for s in vis_ls:
        pos = graphviz_layout(bn, root=root, prog=s)
        nx.draw(
            bn, pos, with_labels=True, arrowsize=40, node_size=2000, alpha=0.8, font_weight="bold"
        )
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


def get_mb_as_bn(model: BayesianNetwork, center: str):
    mb_list = model.get_markov_blanket(center)
    mb = copy.deepcopy(model)

    for n in model.nodes:
        if n not in center.join(mb_list):
            mb.remove_node(n)

    return mb
