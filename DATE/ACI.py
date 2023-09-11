import datetime
import os
import sys

import scipy.stats as stats
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

initial_batch_size = 12
initial_next_batch = 13

# Alternatively create a pointer for each batch, this would be smarter...
splits = samples.groupby('batch_size')
records = {}  #
pv_per_batch_size = {}

for group_name, group_df in splits:
    records[group_name] = group_df
    pv_per_batch_size = pv_per_batch_size | {group_name: (group_name / 30 * 100)}

global_i = 0
current_batch_size = None
next_batch_size = initial_batch_size
next_batch = None
entire_training_data = None
past_training_data = None
sample_size_history = []
avg_surprise_history = []
last_correct = False

def load_next_batch():
    global next_batch
    global next_batch_size
    global current_batch_size
    global global_i

    current_batch_size = next_batch_size

    if global_i + current_batch_size > len(records[current_batch_size]):
        print("Reached the end of the records")
        print(f"Current batch size: {current_batch_size}")
        return False

    next_batch = records[current_batch_size].iloc[global_i:global_i + current_batch_size]
    global_i += current_batch_size
    sample_size_history.append(current_batch_size)
    return True


# print(np.abs(np.mean(samples.iloc[0:10]['distance'].values) - np.mean(samples.iloc[200:205]['distance'].values)))


# print("Entire SD: ", np.std(records[30]['part_delay']))
# print("Entire Mean: ", np.mean(records[30]['part_delay']))
# print(np.percentile(records[30]['part_delay'], 75))


def plot_histogram_with_normal_distribution(column_data):
    plt.hist(column_data, bins=20, density=True, alpha=0.6, color='b', label='Histogram')

    # Fit a normal distribution to the data
    mu, std = norm.fit(column_data)

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


def plot_boxplot(column_data):
    plt.boxplot(column_data)
    plt.show()


def SLOs_fulfilled(batch):
    if batch['part_delay'].sum() < 7000 and np.percentile(batch['distance'], 20) > 6:
        return True
    else:
        return False


def get_surprise(historical_data, batch):
    mean = np.mean(historical_data)
    std_dev = np.std(historical_data)
    pdf_values = stats.norm.pdf(batch, loc=mean, scale=std_dev)
    nll_values = -np.log(np.maximum(pdf_values, 1e-10))
    return np.sum(nll_values)


# plot_histogram_with_normal_distribution(records[30]['part_delay'])


sr_per_batch_size = {}
ig_per_batch_size = {}
surprise_history_per_batch = my_dict = {i: [] for i in range(12, 31)}

while load_next_batch() is not False:

    past_training_data = entire_training_data
    entire_training_data = pd.concat([past_training_data, next_batch])
    # plot_histogram_with_normal_distribution(next_batch['part_delay'])
    # plot_boxplot(next_batch['part_delay'])

    if len(model.get_cpds()) == 0:
        model.fit(data=next_batch)
        next_batch_size = initial_next_batch
        continue

    # Make a prediction regarding the next batch's distribution
    past_data_column = past_training_data[past_training_data['batch_size'] == current_batch_size]['part_delay'].values

    if len(past_data_column):
        avg_surprise_for_observing_batch = get_surprise(past_data_column, next_batch['part_delay']) / current_batch_size

        surprise_history_per_batch[current_batch_size].append(avg_surprise_for_observing_batch)
        # print(avg_surprise_for_observing_batch)
        # print("---------------")

    total_sum, total_count = (0, 0)
    for key, value in my_dict.items():
        total_sum += sum(value)
        total_count += len(value)

    if total_count > 0:
        avg_surprise_over_all_batches = total_sum / total_count
        avg_surprise_history.append(avg_surprise_over_all_batches)
    else:
        avg_surprise_over_all_batches = 5

    for (i, s) in splits:
        split_known = entire_training_data[entire_training_data['batch_size'] == i]

        if len(surprise_history_per_batch[i]) == 0:
            ig_per_batch_size[i] = 115
        else:
            ig_per_batch_size[i] = np.ceil(
                np.median(surprise_history_per_batch[i]) / avg_surprise_over_all_batches * 100)

        interpolation_point = None
        closest_index = initial_batch_size

        # If there is no data yet, rather get the one of the closest known distribution
        if len(split_known) == 0:
            interpolation_point = initial_next_batch
            for j, group_df in entire_training_data.groupby('batch_size'):
                if np.abs(j - i) < np.abs(closest_index - i):
                    interpolation_point = closest_index
                    closest_index = j
            split_known = entire_training_data[entire_training_data['batch_size'] == closest_index]

        valid_items = sum(1 for (x, row) in split_known.iterrows() if row['distance']
                          >= 6 and row['part_delay'] * i <= 7000)
        percentage_failing_SLOs = 100 * (valid_items / len(split_known))

        if interpolation_point is None:
            sr_per_batch_size = sr_per_batch_size | {i: percentage_failing_SLOs}
        else:
            split_known = entire_training_data[entire_training_data['batch_size'] == interpolation_point]
            valid_items = sum(1 for (x, row) in split_known.iterrows() if row['distance']
                              >= 6 and row['part_delay'] * i <= 7000)
            interpolation_point_sr = 100 * (valid_items / len(split_known))

            x1, y1 = (interpolation_point, interpolation_point_sr)
            x2, y2 = (closest_index, percentage_failing_SLOs)

            y = y1 + (i - x1) * (y2 - y1) / (x2 - x1)
            sr_per_batch_size = sr_per_batch_size | {i: y}

    # TODO: Required for comparison later
    if SLOs_fulfilled(next_batch):  # if SLOs are fulfilled and I attribute a high epistemic value to change
        if last_correct:
            next_batch_size += 1
            last_correct = False
        else:
            last_correct = True
    else:
        if not last_correct:
            next_batch_size -= 1
            last_correct = True
        else:
            last_correct = False

    # for (i, s) in splits:
    #     if (sr_per_batch_size[i] * pv_per_batch_size[i] * ig_per_batch_size[i] >
    #             sr_per_batch_size[next_batch_size] * pv_per_batch_size[next_batch_size] * ig_per_batch_size[
    #                 next_batch_size]):
    #         next_batch_size = i

    # print(np.abs(np.mean(next_batch['distance']) - np.mean(past_training_data['distance'])))

    model.fit(data=entire_training_data)
    # var_el = VariableElimination(model)
    # print(var_el.query(variables=["distance"]))

x_values = range(1, len(sample_size_history) + 1)
plt.plot(x_values, sample_size_history, marker='o', linestyle='-')
plt.show()

sys.exit()
