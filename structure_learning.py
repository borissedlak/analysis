import os
import sys

import pandas as pd
import pgmpy.base.DAG
from pgmpy.estimators import K2Score, HillClimbSearch
from pgmpy.readwrite import XMLBIFWriter

import util
from util import print_BN

ROOT = os.path.dirname(__file__)
files = [
    ROOT + f'/data/nano_cpu.csv',
    ROOT + f'/data/xavier_cpu_2_10.csv',
    ROOT + f'/data/xavier_cpu_4_15.csv',
    ROOT + f'/data/xavier_cpu_6_20.csv',
    ROOT + f'/data/xavier_gpu_2_10.csv',
    ROOT + f'/data/xavier_gpu_4_15.csv',
    ROOT + f'/data/xavier_gpu_6_20.csv',
]
samples = pd.concat((pd.read_csv(f) for f in files))

# samples = get_base_data_as_sample()

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

# samples['distance_SLO'] = pd.cut(samples['distance'], bins=[0, 25, max(samples['distance'])],
#                                  labels=[True, False], include_lowest=True)
# Must be a little bit increased in order to allow higher fps
# samples['time_SLO'] = samples['execution_time'] <= (1000 / samples['fps'])

del samples['timestamp']
del samples['cpu_utilization']
del samples['memory_usage']
del samples['device_type']
# del samples['bitrate']
# del samples['pixel']

# Remove unnecessary variables
# del samples['memory_usage']
# del samples['cpu_temperature']

# print(samples.head())

# 1. Learning Structure

scoring_method = K2Score(data=samples)
estimator = HillClimbSearch(data=samples)

dag: pgmpy.base.DAG = estimator.estimate(
    scoring_method=scoring_method, max_indegree=4
)

# Removing wrong edges
dag.remove_edge("pixel", "GPU")  # Simply wrong
dag.remove_edge("bitrate", "config")  # Simply wrong
dag.remove_edge("success", "GPU")  # Simply wrong
dag.remove_edge("success", "execution_time")  # Correlated but not causal

# Reversing edges
dag = util.fix_edge_between_u_v(dag, "GPU", "execution_time")
dag = util.fix_edge_between_u_v(dag, "fps", "bitrate")
dag = util.fix_edge_between_u_v(dag, "config", "consumption")
dag = util.fix_edge_between_u_v(dag, "config", "GPU")

print_BN(dag, vis_ls=["circo"])
print_BN(util.get_mb_as_bn(model=dag, center="bitrate"), root="bitrate", save=True)
print_BN(util.get_mb_as_bn(model=dag, center="distance"), root="distance", save=True)
print_BN(util.get_mb_as_bn(model=dag, center="success"), root="success", save=True)
print_BN(util.get_mb_as_bn(model=dag, center="consumption"), root="consumption", save=True)
print_BN(util.get_mb_as_bn(model=dag, center="execution_time"), root="time_SLO", save=True)

sys.exit()

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

print("Parameter Learning Finished")

# infer_non_adjust = VariableElimination(model)
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

XMLBIFWriter(model).write_xmlbif('model.xml')
print("Model exported as 'model.xml'")

sys.exit()
