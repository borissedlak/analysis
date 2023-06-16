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


# 2) Get the relative number of SLO violations

def compare_samples_SLOs(ideal, naive, distance_slo, success_threshold, time_threshold):
    print(f"success SLO > {success_threshold}")
    success_true = ideal[ideal['transformed']].size
    print("Ideal", '%.2f' % (success_true / ideal.size))
    success_true = naive[naive['transformed']].size
    print("Naive", '%.2f' % (success_true / naive.size), "\n")

    print(f"distance SLO > 0.95 at {distance_slo}")
    distance_true = ideal[ideal[distance_slo]].size
    print("Ideal", '%.2f' % (distance_true / ideal.size))
    distance_true = naive[naive[distance_slo]].size
    print("Naive", '%.2f' % (distance_true / naive.size), "\n")

    print(f"time SLO > {time_threshold}")
    time_true = ideal[samples_ideal['time_slo']].size
    print("Ideal", '%.2f' % (time_true / ideal.size))
    time_true = naive[naive['time_slo']].size
    print("Naive", '%.2f' % (time_true / naive.size), "\n")

    print("Ideal", '%.1f' % (ideal['consumption'].mean()))
    print("Naive", '%.1f' % (naive['consumption'].mean()), "\n")
    print("-----------------------------------------------------\n")


compare_samples_SLOs(samples_ideal, samples_naive, "distance_SLO_hard", success_threshold=0.90, time_threshold=0.95)

# Scenario B
# Model result: 102240 16FPS 2C_10W GPU
# Naive assumption: 57600 26FPS 4C_15W

samples_ideal = util.get_prepared_base_samples(f'/data/xavier_gpu_2_10.csv')
samples_ideal = samples_ideal[(samples_ideal['pixel'] == 102240) & (samples_ideal['fps'] == 16)]

samples_naive = util.get_prepared_base_samples(f'/data/xavier_gpu_4_15.csv')
samples_naive = samples_naive[(samples_naive['pixel'] == 57600) & (samples_naive['fps'] == 30)]  # 20 | 26 missing!!!

compare_samples_SLOs(samples_ideal, samples_naive, "distance_SLO_easy", success_threshold=0.98, time_threshold=0.75)

sys.exit()
