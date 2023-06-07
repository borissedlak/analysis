import os
import sys

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator, K2Score, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

from util import print_BN

ROOT = os.path.dirname(__file__)

# model = get_example_model("alarm")
# samples = BayesianModelSampling(model).forward_sample(size=int(1e2), seed=35)
files = [
         # ROOT + f'/data/nano_cpu.csv',
         ROOT + f'/data/xavier_cpu_2_10.csv',
         ROOT + f'/data/xavier_cpu_4_15.csv',
         ROOT + f'/data/xavier_cpu_6_20.csv',
]

samples = pd.concat((pd.read_csv(f) for f in files))

# samples['device_type'] = 'Xavier NX'
# samples['GPU'] = True
# samples['config'] = "6C 20W"
# samples.to_csv(ROOT + f'/data/nano_cpu.csv', encoding='utf-8', index=False)
# sys.exit()


# Sanity check
# print(samples.isna().any())
samples = samples[samples['consumption'].notna()]

samples['distance'] = samples['distance'].astype(int)

samples['cpu_pods'] = pd.cut(samples['cpu_utilization'], bins=[0, 50, 70, 90, 100],
                             labels=['Low', 'Mid', 'High', 'Very High'], include_lowest=True)
samples['memory_pods'] = pd.cut(samples['memory_usage'], bins=[0, 50, 70, 90, 100],
                                labels=['Low', 'Mid', 'High', 'Very High'], include_lowest=True)

samples['bitrate'] = samples['fps'] * samples['pixel']

# If this is on 30, it still includes 12 FPS
samples['distance_SLO'] = pd.cut(samples['distance'], bins=[0, 25, max(samples['distance'])],
                                 labels=[True, False], include_lowest=True)
# Must be a little bit increased in order to allow higher fps
samples['time_SLO'] = samples['execution_time'] <= (1000 / samples['fps'])


del samples['timestamp']
del samples['cpu_utilization']
del samples['memory_usage']
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

model = BayesianNetwork(ebunch=dag.edges())
model.fit(data=samples, estimator=MaximumLikelihoodEstimator)
# print(trained_mle.get_cpds("success"))

# trained_mle_2 = BayesianNetwork(ebunch=dag.edges())
# trained_mle_2.fit(data=samples, estimator=MaximumLikelihoodEstimator)
#
# trained_mle_3 = BayesianNetwork(ebunch=dag.edges())
# trained_mle_3.fit(
#     samples, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=0.1
# )

print("Parameter Learning Finished")

# 3. Causal Inference

infer_non_adjust = VariableElimination(model)
# print(infer_non_adjust.query(variables=["time_SLO"]))
# print(infer_non_adjust.query(variables=["success"],
#                              evidence={'within_time': True, 'fps': 60}))

bitrate_list = model.get_cpds("bitrate").__getattribute__("state_names")["bitrate"]
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
