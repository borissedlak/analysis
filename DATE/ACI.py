import datetime
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader
from scipy.stats import norm

from util import get_prepared_base_samples

ROOT = os.path.dirname(__file__)
samples = pd.read_csv(ROOT + '/refined.csv')
model = XMLBIFReader("../model.xml").get_model()

# Alternatively create a pointer for each batch, this would be smarter...
splits = samples.groupby('batch_size')
records = {}  #

for group_name, group_df in splits:
    records[group_name] = group_df

global_i = 0
current_batch_size = 30
next_batch = None
entire_training_data = None
past_training_data = None


def load_next_batch():
    global next_batch
    global global_i

    if global_i + current_batch_size > len(records[current_batch_size]):
        print("Reached the end of the records")
        return False

    next_batch = records[current_batch_size].iloc[global_i:global_i + current_batch_size]
    global_i += current_batch_size
    return True


# print(np.abs(np.mean(samples.iloc[0:10]['distance'].values) - np.mean(samples.iloc[200:205]['distance'].values)))


print("Entire SD: ", np.std(records[30]['part_delay']))
print("Entire Mean: ", np.mean(records[30]['part_delay']))
print(np.percentile(records[30]['part_delay'], 75))


def plot_histogram_with_normal_distribution(data):
    plt.hist(data, bins=20, density=True, alpha=0.6, color='b', label='Histogram')

    # Fit a normal distribution to the data
    mu, std = norm.fit(data)

    # Create a range of values for the x-axis
    xmin, xmax = plt.xlim()
    x = np.linspace(0, xmax, 100)

    # Calculate the probability density function (PDF) for the normal distribution
    p = norm.pdf(x, mu, std)

    # Plot the normal distribution curve
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')

    # Add labels and a legend
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram and Normal Distribution')
    plt.legend()

    # Show the plot
    plt.show()

plot_histogram_with_normal_distribution(records[30]['part_delay'])

while load_next_batch() is not False:

    past_training_data = entire_training_data
    entire_training_data = pd.concat([past_training_data, next_batch])
    plot_histogram_with_normal_distribution(next_batch['part_delay'])

    plt.boxplot(next_batch['part_delay'])
    plt.show()

    if len(model.get_cpds()) == 0:
        model.fit(data=next_batch)
        continue

    # print(np.abs(np.mean(next_batch['distance']) - np.mean(past_training_data['distance'])))

    model.fit(data=entire_training_data)
    # var_el = VariableElimination(model)
    # print(var_el.query(variables=["distance"]))

sys.exit()
