import csv
import os
import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

plt.rcParams['font.family'] = 'serif'

ROOT = os.path.dirname(__file__)
csv_filename = ROOT + '/regression_points.csv'

# Read the data from the CSV file into an array of tuples
data = []
with open(csv_filename, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append((int(row[0]), int(row[1])))

data = sorted(data, key=lambda x: x[0])

x_utilization = np.array([item[0] for item in data])
y_delay_per_part = np.array([item[1] for item in data])

poly_features = PolynomialFeatures(degree=2)
x_poly = poly_features.fit_transform(x_utilization.reshape(-1, 1))

model_full_training_data = LinearRegression()
model_full_training_data.fit(x_poly, np.array(y_delay_per_part))
x_range = np.linspace(38, 85, 100)  # Adjust the number of points as needed
y_pred_full = model_full_training_data.predict(poly_features.fit_transform(x_range.reshape(-1, 1)))

x_poly = poly_features.fit_transform(x_utilization[:40].reshape(-1, 1))

model_part_training_data = LinearRegression()
model_part_training_data.fit(x_poly, np.array(y_delay_per_part[:40]))
x_range = np.linspace(38, 85, 100)  # Adjust the number of points as needed
y_pred_part = model_part_training_data.predict(poly_features.fit_transform(x_range.reshape(-1, 1)))

fig, ax = plt.subplots()
ax.scatter(x_utilization, y_delay_per_part, label='Observations', marker='o')
ax.plot(x_range, y_pred_full, label='Full Data', color='red')
ax.plot(x_range, y_pred_part, label='Thirty Values', color='green')

# Add labels and a legend
ax.set_xlabel('Utilization')
ax.set_ylabel('Delay / Part')
ax.set_title('Polynomial relation between utilization and part_delay')
ax.legend()

ax.grid(True)
fig.set_size_inches(6, 4)
plt.savefig("relation_batch_size.png", dpi=400, bbox_inches="tight")  # default dpi is 100
plt.show()


# Now second figure


y_pred_part = model_full_training_data.predict(poly_features.fit_transform(x_utilization.reshape(-1, 1)))
y_pred_full = model_full_training_data.predict(poly_features.fit_transform(x_utilization.reshape(-1, 1)))

diff_part = []
diff_full = []

for i in range(len(x_utilization)-1):
    diff_part.append(np.abs(y_delay_per_part[i] - y_pred_part[i]))
    diff_full.append(np.abs(y_delay_per_part[i] - y_pred_full[i]))

fig, ax = plt.subplots()

