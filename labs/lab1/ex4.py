import numpy as np
from sklearn.metrics import confusion_matrix

rng = np.random.default_rng()

n = 1000
contamination = 0.1
n_out = int(n * contamination)

mu = np.array([1.0, -2.0, 3.0])
Sigma = np.array([
    [4.0, 1.2, 0.5],
    [1.2, 2.0, 0.3],
    [0.5, 0.3, 1.0],
])
d = len(mu)

L = np.linalg.cholesky(Sigma)

x = rng.standard_normal((n, d))

y_true = np.zeros(n, dtype=int)
y_true[:n_out] = 1
x[:n_out] *= 3

data = x @ L.T + mu

mean_hat = data.mean(axis=0)
cov_hat = np.cov(data, rowvar=False)
L_hat = np.linalg.cholesky(cov_hat)
z = np.linalg.solve(L_hat, (data - mean_hat).T).T
scores = np.linalg.norm(z, axis=1)


threshold = np.quantile(scores, 1 - contamination)
y_pred = (scores > threshold).astype(int)

print("Threshold:", threshold)
print("Predicted anomalies:", y_pred.sum(), "| True anomalies:", y_true.sum())


tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
ba = (tpr + tnr) / 2

print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"TPR={tpr:.3f} TNR={tnr:.3f} BA={ba:.3f}")