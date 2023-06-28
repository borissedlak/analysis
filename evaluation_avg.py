import itertools
import os
import statistics
import sys

import pandas as pd

import util

# Scenario A
# Model result: 102240 20FPS 4C_15W NO-GPU
# Naive assumption: 230400 30FPS 6C_20W

comb = list(itertools.product([25440, 102240, 230400, 409920, 921600], [12, 16, 20, 26, 30],  # 57600 missing!!
                              ['xavier_cpu_2_10', 'xavier_cpu_4_15',
                               'xavier_cpu_6_20']))#, 'xavier_gpu_2_10',
                               #'xavier_gpu_4_15', 'xavier_gpu_6_20']))

success_true = list()
distance_true = list()
distance_abs = list()
time_true = list()
consumption = list()

for (pixel, fps, conf) in comb:
    samples = util.get_prepared_base_samples(f'/data/{conf}.csv')
    samples = samples[(samples['pixel'] == pixel) & (samples['fps'] == fps)]

    success_true.append(samples[samples['transformed']].size / samples.size)
    distance_true.append(samples[samples["distance_SLO_easy"]].size / samples.size)  # distance_SLO_easy
    distance_abs.append(samples['distance'].mean())
    time_true.append(samples[samples["time_slo"]].size / samples.size)
    consumption.append(samples['consumption'].mean())

print("Transform:", statistics.mean(success_true))
print("Distance (Rel):", statistics.mean(distance_true))
print("Distance (Abs):", statistics.mean(distance_abs))
print("Time:", statistics.mean(time_true))
print("Consumption:", statistics.mean(consumption))

px = [25440, 57500, 102240, 230400, 409920, 921600]
fps = [12, 16, 20, 26, 30]

print(statistics.mean(px) * statistics.mean(fps))

sys.exit()
