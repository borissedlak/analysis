import copy

import numpy as np
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

original_model = get_example_model("alarm")
samples = BayesianModelSampling(original_model).forward_sample(size=int(1e3), seed=30)
# # print(samples.head())
#
# learned_model = BayesianNetwork(ebunch=original_model.edges())
# # print(model_struct.nodes())
# # print(alarm_model.nodes())
#
# # Fitting the model using Maximum Likelihood Estimator
#
# mle = MaximumLikelihoodEstimator(model=learned_model, data=samples)
#
# # Estimating the CPD for a single node.
# print(mle.estimate_cpd(node="FIO2"))
# print(mle.estimate_cpd(node="CVP"))
#
# # Estimating CPDs for all the nodes in the model
# val = mle.get_parameters()
#
# # Verifying that the learned parameters are almost equal.
# print(np.allclose(
#     original_model.get_cpds("FIO2").values, mle.estimate_cpd("FIO2").values, atol=0.01
# ))


trained_mle = BayesianNetwork(ebunch=original_model.edges())
trained_mle.fit(data=samples, estimator=MaximumLikelihoodEstimator)
print(trained_mle.get_cpds("FIO2"))

trained_be = BayesianNetwork(ebunch=original_model.edges())
trained_be.fit(
    data=samples,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=100,
)
print(trained_be.get_cpds("FIO2"))

mb_list = trained_mle.get_markov_blanket('VENTALV')
mb = copy.deepcopy(trained_mle)
mb.remove_node("CO")

for n in trained_mle.nodes:
    if n not in 'VENTALV'.join(mb_list):
        mb.remove_node(n)


for s in ['dot', 'twopi', 'fdp', 'sfdp', 'circo']:
    pos = graphviz_layout(mb, prog=s, root="VENTALV")
    nx.draw(
        mb, pos,
        with_labels=True, arrowsize=30, node_size=1500, alpha=0.8, font_weight="bold"
    )
    plt.show()
