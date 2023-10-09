import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=1)

model_Laptop = joblib.load("models/Laptop.sav")
model_Xavier = joblib.load("models/Xavier.sav")

x_range = np.linspace(1, 16)
fig, ax = plt.subplots()

for m, gpu, s, c in [(model_Laptop, 0, "Laptop CPU", "red"), (model_Xavier, 0, "Xavier CPU", "green"),
                     (model_Xavier, 1, "Xavier GPU", "blue")]:
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
ax.set_xlim(1, 15)
ax.set_xticks(range(1, 17, 2))

ax.grid(True)
fig.set_size_inches(5.5, 3.3)
plt.savefig("regression_slo_stream.png", dpi=600, bbox_inches="tight")  # default dpi is 100
plt.show()
