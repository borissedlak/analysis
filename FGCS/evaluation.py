import itertools
import os
import sys

import pandas as pd

import util

# Scenario A
# Model result: 102240 20FPS 4C_15W NO-GPU
# Naive assumption: 230400 30FPS 6C_20W


# 1) Load all the data

# Configuration parameters

pixel = 102240
fps = 26

samples_nano = util.get_prepared_base_samples(f'/data/nano_cpu.csv', train=False)
samples_nano = samples_nano[(samples_nano['pixel'] == pixel) & (samples_nano['fps'] == fps)]

samples_xavier_cpu_2_10 = util.get_prepared_base_samples(f'/data/xavier_cpu_2_10.csv', train=False)
samples_xavier_cpu_2_10 = samples_xavier_cpu_2_10[(samples_xavier_cpu_2_10['pixel'] == pixel) & (samples_xavier_cpu_2_10['fps'] == fps)]

samples_xavier_gpu_2_10 = util.get_prepared_base_samples(f'/data/xavier_gpu_2_10.csv', train=False)
samples_xavier_gpu_2_10 = samples_xavier_gpu_2_10[(samples_xavier_gpu_2_10['pixel'] == pixel) & (samples_xavier_gpu_2_10['fps'] == fps)]

samples_laptop = util.get_prepared_base_samples(f'/data/laptop_cpu.csv', train=False)
samples_laptop = samples_laptop[(samples_laptop['pixel'] == pixel) & (samples_laptop['fps'] == fps)]


# 2) Get the relative number of SLO violations

def compare_samples_SLOs(pairs, distance_slo, success_threshold, time_threshold):

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


compare_samples_SLOs([("Nano", samples_nano), ("CPU 2C 10W", samples_xavier_cpu_2_10),
                      ("GPU 2C 10W", samples_xavier_gpu_2_10), ("Laptop", samples_laptop)], "distance_SLO_hard",
                     success_threshold=0.90, time_threshold=0.95)

sys.exit()
