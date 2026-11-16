import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.utils.utility import standardizer
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.dif import DIF

MAT_FILE = "shuttle.mat"   
TEST_SIZE = 0.4            
N_SPLITS = 10              


#1.
def load_shuttle(path=MAT_FILE):
    data = loadmat(path)                     
    return data["X"], data["y"].ravel().astype(int)


def split_and_normalise(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed,
        stratify=y)                          
    
    X_train_n, X_test_n = standardizer(X_train, X_test)
    return X_train_n, X_test_n, y_train, y_test


def make_models(contamination):
    
    return {
        "IForest": IForest(contamination=contamination, random_state=0),   
        "LODA": LODA(contamination=contamination),                        
        "DIF": DIF(contamination=contamination, random_state=0),           
    }


#2.
if __name__ == "__main__":
    X, y = load_shuttle()
    print("X:", X.shape, " anomaly ratio: %.4f" % y.mean())
    results = {n: {"BA": [], "AUC": []} for n in ["IForest", "LODA", "DIF"]}

    for seed in range(N_SPLITS):
        X_tr, X_te, y_tr, y_te = split_and_normalise(X, y, seed)
        contamination = float(y_tr.mean())   
        for name, model in make_models(contamination).items():
            model.fit(X_tr)                  
            ba = balanced_accuracy_score(y_te, model.predict(X_te))     
            auc = roc_auc_score(y_te, model.decision_function(X_te))    
            results[name]["BA"].append(ba); results[name]["AUC"].append(auc)
            print(f"split {seed} {name:8s} BA={ba:.4f} AUC={auc:.4f}")

    print(f"\n=== mean over {N_SPLITS} splits ===")
    for name, r in results.items():
        print(f"{name:8s} BA = {np.mean(r['BA']):.4f} (+-{np.std(r['BA']):.4f})   "
              f"ROC AUC = {np.mean(r['AUC']):.4f} (+-{np.std(r['AUC']):.4f})")