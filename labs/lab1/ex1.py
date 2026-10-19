import matplotlib.pyplot as plt
from pyod.utils.data import generate_data

X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=0.1
)

inliers = X_train[y_train == 0]
outliers = X_train[y_train == 1]

plt.figure(figsize=(20, 20))
plt.scatter(inliers[:, 0], inliers[:, 1], c="tab:blue", label="Inliers")
plt.scatter(outliers[:, 0], outliers[:, 1], c="tab:red", label="Outliers")
plt.title("Training samples")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()