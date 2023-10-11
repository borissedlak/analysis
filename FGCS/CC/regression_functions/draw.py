import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=1)

model_Laptop = joblib.load("models/Laptop.sav")
model_Xavier = joblib.load("models/Xavier.sav")
model_Orin = joblib.load("models/Orin.sav")
model_Nano = joblib.load("models/Nano.sav")

# y_pred_full = model_Xavier.predict(poly_features.fit_transform(np.array([[1, 0]])))
# y_pred_factor = y_pred_full[:, 0] * y_pred_full[:, 1]
# print(y_pred_factor)
#
# y_pred_full = model_Xavier.predict(poly_features.fit_transform(np.array([[5, 1]])))
# y_pred_factor = y_pred_full[:, 0] * y_pred_full[:, 1]
# print(y_pred_factor)
#
# y_pred_full = model_Laptop.predict(poly_features.fit_transform(np.array([[9, 0]])))
# y_pred_factor = y_pred_full[:, 0] * y_pred_full[:, 1]
# print(y_pred_factor)
#
# y_pred_full = model_Orin.predict(poly_features.fit_transform(np.array([[9, 1]])))
# y_pred_factor = y_pred_full[:, 0] * y_pred_full[:, 1]
# print(y_pred_factor)
#
# y_pred_full = model_Nano.predict(poly_features.fit_transform(np.array([[1, 0]])))
# y_pred_factor = y_pred_full[:, 0] * y_pred_full[:, 1]
# print(y_pred_factor)
#
# sys.exit()

x_range = np.linspace(1, 26)
fig, ax = plt.subplots()

for m, gpu, s, c in [(model_Laptop, 0, r'$\mathit{Laptop}$', "firebrick"),
                     (model_Xavier, 0, r'$\mathit{Xavier_{CPU}}$', "mediumaquamarine"),
                     (model_Xavier, 1, r'$\mathit{Xavier_{GPU}}$', "steelblue"),
                     (model_Orin, 1, r'$\mathit{Orin}$', "dimgray"),
                     (model_Nano, 0, r'$\mathit{Nano}$', "chocolate")]:
    input_data = np.column_stack((x_range, np.full(x_range.shape, gpu)))
    y_pred_full = m.predict(poly_features.fit_transform(input_data))
    y_pred_factor = y_pred_full[:, 0] * y_pred_full[:, 1]
    # y_pred_factor = np.maximum(y_pred_factor, 0)
    # ax.scatter(x_range, y_pred_full, label='Observations', marker='o')
    ax.plot(x_range, y_pred_factor, label=s, color=c)
    # ax.plot(x_range, y_pred_part, label='Thirty Values', color='red')

# Add labels and a legend
ax.set_xlabel('Number of Streams Processed')
ax.set_ylabel('Estimated (PV x RA)')
# ax.set_title('Polynomial relation between utilization and part_delay')
ax.legend()
ax.set_ylim(0.0, 1.05)
ax.set_xlim(1, 20)
ax.set_xticks(range(1, 25, 3))

ax.grid(True)
fig.set_size_inches(5.5, 3.3)
plt.savefig("regression_slo_stream.png", dpi=600, bbox_inches="tight")  # default dpi is 100
plt.show()
