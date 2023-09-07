import os
import sys

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter

import util

# 0. Preprocessing

ROOT = os.path.dirname(__file__)
samples = pd.read_csv(ROOT + '/laptop_cpu.csv')

# Remove the rows after the one which were false because they have excessive distance

rows_no_success = samples[samples['success'] == False]
# print(len(rows_no_success))
occurrence_indexes = rows_no_success.index.tolist()
for index in occurrence_indexes:
    next_index = index + 1
    if next_index in samples.index:
        samples.drop(next_index, inplace=True)

samples.rename(columns={'execution_time': 'part_delay', 'cpu_utilization': 'utilization', 'fps': 'batch_size'},
               inplace=True)

del samples['timestamp']
del samples['memory_usage']
del samples['success']
del samples['pixel']

# samples['utilization'] = pd.cut(samples['utilization'], bins=7)
# samples['part_delay'] = pd.cut(samples['part_delay'], bins=10)

samples = samples.sample(frac=1, random_state=1)
samples = samples.sort_values(by='batch_size')

samples.to_csv(ROOT + '/refined.csv', index=False)
ROOT = os.path.dirname(__file__)
samples = pd.read_csv(ROOT + '/refined.csv')

# 1. Learning Structure

model = BayesianNetwork([('batch_size', 'distance'), ('batch_size', 'utilization'), ('utilization', 'part_delay'),
                         ('batch_size', 'batch_delay'), ('part_delay', 'batch_delay')])

util.print_BN(model, vis_ls=["circo"], root="batch_size", save=True, show=False,
              color_map=[util.regular, util.regular, util.regular, util.regular, util.regular])

model.remove_node("batch_delay")

print("Structure Learning Finished")

# 2. Learning Parameters
# model.fit(data=samples, estimator=MaximumLikelihoodEstimator)


XMLBIFWriter(model).write_xmlbif('model.xml')
print("Model exported as 'model.xml'")

sys.exit()
