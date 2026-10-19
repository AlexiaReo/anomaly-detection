import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix, roc_curve

X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=0.1,
)


def report(y_true, y_pred, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    ba = (tpr + tnr) / 2
    print(f"{name}: TN={tn} FP={fp} FN={fn} TP={tp} | TPR={tpr:.3f} TNR={tnr:.3f} BA={ba:.3f}")


model = IForest(contamination=0.1)
model.fit(X_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

report(y_train, y_train_pred, "Train")
report(y_test, y_test_pred, "Test ")

y_test_scores = model.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)

plt.figure(figsize=(20, 20))
plt.plot(fpr, tpr)
plt.title("ROC curve (test data)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

for c in [0.01, 0.05, 0.1, 0.2, 0.3]:
    m = IForest(contamination=c)
    m.fit(X_train)
    print(f"\ncontamination = {c}")
    report(y_train, m.predict(X_train), "Train")
    report(y_test, m.predict(X_test), "Test ")