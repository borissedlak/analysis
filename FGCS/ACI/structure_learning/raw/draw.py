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

import util

# Create a directed graph
bn = nx.DiGraph()

nodes = ['bitrate', 'in_time', 'success', 'fps', 'pixel', 'streams', 'distance', 'CPU', 'consumption', 'network', 'memory', 'pixel']

for n in nodes:
    bn.add_node(n)

bn.add_edge("fps", "distance")
bn.add_edge("CPU", "consumption")

pos = graphviz_layout(bn)
nx.draw(
    bn, pos, with_labels=True, arrowsize=20, node_size=1500,
)
plt.box(False)
plt.savefig(f"abcd.png", dpi=400, bbox_inches="tight")  # default dpi is 100
plt.show()

# Draw the graph
# pos = nx.spring_layout(G)  # Position nodes using a spring layout
# nx.draw(G, pos, with_labels=True, node_size=1500, node_color='skyblue', font_color='black', arrows=True)
# plt.title("Directed Acyclic Graph (DAG)")
# plt.axis('off')  # Turn off axis
# plt.show()

# Optional: Save the DAG as an image
# plt.savefig("dag.png", format="PNG")