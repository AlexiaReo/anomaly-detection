import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix

contamination = 0.1

X, y = generate_data(
    n_train=1000,
    n_features=1,
    contamination=contamination,
    train_only=True,
)
x = X.ravel()

z = (x - x.mean()) / x.std()
abs_z = np.abs(z)

threshold = np.quantile(abs_z, 1 - contamination)
y_pred = (abs_z > threshold).astype(int)

print("Threshold:", threshold)
print("Predicted anomalies:", y_pred.sum(), "| True anomalies:", int(y.sum()))

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
ba = (tpr + tnr) / 2

print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"TPR={tpr:.3f} TNR={tnr:.3f} BA={ba:.3f}")