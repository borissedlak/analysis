import os
import sys

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator, TreeSearch, K2Score, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling

from util import print_BN

ROOT = os.path.dirname(__file__)

# model = get_example_model("alarm")
# samples = BayesianModelSampling(model).forward_sample(size=int(1e2), seed=35)
files = [ROOT + f'/data/nano_cpu.csv']

samples = pd.concat((pd.read_csv(f) for f in files))

# Sanity check
print(samples.isna().any())

samples = samples[samples['consumption'].notna()]

samples['distance'] = samples['distance'].astype(int)

# samples['cpu_pods'] = pd.cut(samples['cpu_utilization'], bins=[0, 50, 70, 90, 100],
#                              labels=['Low', 'Mid', 'High', 'Very High'], include_lowest=True)
# samples['memory_pods'] = pd.cut(samples['memory_usage'], bins=[0, 50, 70, 90, 100],
#                                 labels=['Low', 'Mid', 'High', 'Very High'], include_lowest=True)
#
# samples['bitrate'] = samples['fps'] * samples['pixel']
#
# # If this is on 30, it still includes 12 FPS
# samples['distance_SLO'] = pd.cut(samples['distance'], bins=[0, 25, max(samples['distance'])],
#                                  labels=[True, False], include_lowest=True)
# # Must be a little bit increased in order to allow higher fps
# samples['time_SLO'] = samples['execution_time'] <= (1200 / samples['fps'])


# Why would I actually need multiple bins, if I'm only interested in whether the SLO is fulfilled or not
# samples['bitrate_pod'] = pd.cut(samples['memory_usage'], bins=[0, max_value/num_pods, 2*max_value/num_pods, max_value], labels=False)

del samples['timestamp']
# del samples['cpu_utilization']
# del samples['memory_usage']
# del samples['bitrate']
# del samples['pixel']

# Remove unnecessary variables
# del samples['memory_usage']
# del samples['cpu_temperature']

# print(samples.head())

# 1. Learning Structure

scoring_method = K2Score(data=samples)
estimator = HillClimbSearch(data=samples)
dag = estimator.estimate(
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
)
print_BN(dag, vis_ls=["circo"])

# print_BN(get_mb_as_bn(estimated_model, "success"))
# print_BN(get_mb_as_bn(estimated_model, "pixel"))

# est = TreeSearch(samples)
# dag = est.estimate(estimator_type="chow-liu")
# print_BN(dag, try_visualization=False)

# get_f1_score(estimated_model, model)

print("Structure Learning Finished")

# 2. Learning Parameters

trained_mle = BayesianNetwork(ebunch=dag.edges())
trained_mle.fit(data=samples, estimator=MaximumLikelihoodEstimator)
# print(trained_mle.get_cpds("success"))

# trained_mle_2 = BayesianNetwork(ebunch=dag.edges())
# trained_mle_2.fit(data=samples, estimator=MaximumLikelihoodEstimator)
#
# trained_mle_3 = BayesianNetwork(ebunch=dag.edges())
# trained_mle_3.fit(
#     samples, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=0.1
# )

samples = BayesianModelSampling(trained_mle).forward_sample(size=int(63000), seed=30)
samples['execution_time'] = np.floor(samples['execution_time'] * 1.10)
samples['device_type'] = 'Nano 2GB'
samples['GPU'] = False
samples['Config'] = "None"
samples.to_csv(ROOT + f'/data/nano_cpu_2GB.csv', encoding='utf-8', index=False)

print("Parameter Learning Finished")

# 3. Causal Inference

infer_non_adjust = VariableElimination(trained_mle)
# print(infer_non_adjust.query(variables=["time_SLO"]))
# print(infer_non_adjust.query(variables=["success"],
#                              evidence={'within_time': True, 'fps': 60}))

bitrate_list = trained_mle.get_cpds("bitrate").__getattribute__("state_names")["bitrate"]
bitrate_comparison = []

print(infer_non_adjust.query(variables=["time_SLO"]))

for br in bitrate_list:
    sr = infer_non_adjust.query(variables=["distance_SLO", "time_SLO"], evidence={'bitrate': br}).values[1][1]
    if sr > 0.7:
        cons = samples[samples['bitrate'] == br]['consumption'].mean()
        bitrate_comparison.append((br, sr,
                                   samples[samples['bitrate'] == br]['pixel'].iloc[0],
                                   samples[samples['bitrate'] == br]['fps'].iloc[0],
                                   cons))

for (br, sr, pixel, fps, cons) in bitrate_comparison:
    print(pixel, fps, sr, cons)

sys.exit()
