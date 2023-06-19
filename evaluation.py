import os
import sys

import pandas as pd

import util

# Scenario A
# Model result: 102240 20FPS 4C_15W NO-GPU
# Naive assumption: 230400 30FPS 6C_20W


# 1) Load all the data

samples_ideal = util.get_prepared_base_samples(f'/data/xavier_cpu_4_15.csv')
samples_ideal = samples_ideal[(samples_ideal['pixel'] == 102240) & (samples_ideal['fps'] == 20)]

samples_naive = util.get_prepared_base_samples(f'/data/xavier_cpu_6_20.csv')
samples_naive = samples_naive[(samples_naive['pixel'] == 230400) & (samples_naive['fps'] == 30)]

samples_random_1 = util.get_prepared_base_samples(f'/data/xavier_cpu_6_20.csv')
samples_random_1 = samples_random_1[(samples_random_1['pixel'] == 25440) & (samples_random_1['fps'] == 16)]

samples_random_2 = util.get_prepared_base_samples(f'/data/xavier_cpu_2_10.csv')
samples_random_2 = samples_random_2[(samples_random_2['pixel'] == 921600) & (samples_random_2['fps'] == 12)]


# 2) Get the relative number of SLO violations

def compare_samples_SLOs(ideal, naive, random1, random2, distance_slo, success_threshold, time_threshold):

    pairs = [("Ideal", ideal), ("Naive", naive), ("Random #1", random1), ("Random #2", random2)]

    print(f"success SLO > {success_threshold}")
    for (name, variable) in pairs:
        success_true = variable[variable['transformed']].size
        print(name, '%.2f' % (success_true / variable.size))
    print("\n")

    print(f"distance SLO > 0.95 at {distance_slo}")
    for (name, variable) in pairs:
        distance_true = variable[variable[distance_slo]].size
        print(name, '%.2f' % (distance_true / variable.size))
        print(name, " Mean", '%.1f' % (variable['distance'].mean()))
    print("\n")

    print(f"time SLO > {time_threshold}")
    for (name, variable) in pairs:
        time_true = variable[variable['time_slo']].size
        print(name, '%.2f' % (time_true / variable.size))
    print("\n")

    print(f"Consumption")
    for (name, variable) in pairs:
        print(name, '%.1f' % (variable['consumption'].mean()))
    print("\n")
    print("-----------------------------------------------------\n")


compare_samples_SLOs(samples_ideal, samples_naive, samples_random_1, samples_random_2, "distance_SLO_hard", success_threshold=0.90, time_threshold=0.95)

# Scenario B
# Model result: 102240 16FPS 2C_10W GPU
# Naive assumption: 57600 26FPS 4C_15W

samples_ideal = util.get_prepared_base_samples(f'/data/xavier_gpu_2_10.csv')
samples_ideal = samples_ideal[(samples_ideal['pixel'] == 102240) & (samples_ideal['fps'] == 16)]

samples_naive = util.get_prepared_base_samples(f'/data/xavier_gpu_4_15.csv')
samples_naive = samples_naive[(samples_naive['pixel'] == 57600) & (samples_naive['fps'] == 30)]  # 20 | 26 missing!!!

samples_random_1 = util.get_prepared_base_samples(f'/data/xavier_gpu_2_10.csv')
samples_random_1 = samples_random_1[(samples_random_1['pixel'] == 230400) & (samples_random_1['fps'] == 20)]

samples_random_2 = util.get_prepared_base_samples(f'/data/xavier_gpu_6_20.csv')
samples_random_2 = samples_random_2[(samples_random_2['pixel'] == 409920) & (samples_random_2['fps'] == 30)]

compare_samples_SLOs(samples_ideal, samples_naive, samples_random_1, samples_random_2, "distance_SLO_easy", success_threshold=0.98, time_threshold=0.75)

sys.exit()
