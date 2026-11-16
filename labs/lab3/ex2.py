import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

CENTERS_2D = [(10, 0), (0, 10)]            
CENTERS_3D = [(0, 10, 0), (10, 0, 10)]     
STD, N_PER_CLUSTER = 1.0, 500              
CONTAMINATION = 0.02                       
N_TEST, TEST_LOW, TEST_HIGH = 1000, -10, 20  
FIG_DIR = "figures"              # CAN CHANGE: folder where the plots are saved
SEED = 0                                   # CAN CHANGE
DIF_HIDDEN = [[4, 2], [64, 32], [256, 128]]  # CAN CHANGE (point 5; check help(DIF) for your pyod version)
LODA_BINS = [3, 10, 50, 200]                 # CAN CHANGE (point 5)


def plot_score_map(fig, ax, X_test, scores, title, cmap="viridis", s=12):
    """Scatter the test points coloured by their anomaly score (the 'colormap' the lab asks for). 2D or 3D."""
    if X_test.shape[1] == 2:
        sc = ax.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap=cmap, s=s)
        ax.set_aspect("equal", adjustable="box")
    else:
        sc = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=scores, cmap=cmap, s=s)
        ax.set_zlabel("x3")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_title(title)
    fig.colorbar(sc, ax=ax, shrink=0.7)


#1.
def make_train_test(centers, seed=SEED):
    X_train, _ = make_blobs(n_samples=[N_PER_CLUSTER] * len(centers), centers=centers,
                            cluster_std=STD, random_state=seed)
    X_test = np.random.RandomState(seed + 1).uniform(TEST_LOW, TEST_HIGH, size=(N_TEST, len(centers[0])))
    return X_train, X_test


def build_models(dif_hidden=None, loda_bins=10):
    """dif_hidden=None -> pyod default hidden layers."""
    dif_kwargs = {} if dif_hidden is None else {"hidden_neurons": dif_hidden}
    return {
        "IForest": IForest(contamination=CONTAMINATION, random_state=0),   
        "DIF": DIF(contamination=CONTAMINATION, random_state=0, **dif_kwargs),
        "LODA": LODA(contamination=CONTAMINATION, n_bins=loda_bins),
    }


def plot_models(models, X_train, X_test, fname):
    """Fit every model and draw one subplot per model (2D or 3D depending on the data)."""
    dim = X_test.shape[1]
    fig = plt.figure(figsize=(5.5 * len(models), 5))
    for i, (name, m) in enumerate(models.items(), 1):
        m.fit(X_train)
        ax = fig.add_subplot(1, len(models), i, projection="3d" if dim == 3 else None)
        plot_score_map(fig, ax, X_test, m.decision_function(X_test), name)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, fname), dpi=120)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    X_train, X_test = make_train_test(CENTERS_2D)

    #2,3.
    iforest = IForest(contamination=CONTAMINATION, random_state=0).fit(X_train)
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_score_map(fig, ax, X_test, iforest.decision_function(X_test), "IForest (bright = more anomalous)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "ex2_3_iforest.png"), dpi=120)

    #4.
    plot_models(build_models(), X_train, X_test, "ex2_4_three_models.png")

    #5.
    ncol = max(len(DIF_HIDDEN), len(LODA_BINS))
    fig = plt.figure(figsize=(5 * ncol, 10))
    for i, h in enumerate(DIF_HIDDEN, 1):
        m = DIF(contamination=CONTAMINATION, hidden_neurons=h, random_state=0).fit(X_train)
        plot_score_map(fig, fig.add_subplot(2, ncol, i), X_test, m.decision_function(X_test), f"DIF hidden={h}")
    for i, nb in enumerate(LODA_BINS, 1):
        m = LODA(contamination=CONTAMINATION, n_bins=nb).fit(X_train)
        plot_score_map(fig, fig.add_subplot(2, ncol, ncol + i), X_test, m.decision_function(X_test), f"LODA bins={nb}")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "ex2_5_parameters.png"), dpi=120)

    #6.
    X_train3, X_test3 = make_train_test(CENTERS_3D)
    plot_models(build_models(), X_train3, X_test3, "ex2_6_3d_default.png")
    plot_models(build_models(dif_hidden=[4, 2], loda_bins=3), X_train3, X_test3, "ex2_6_3d_small_params.png")
    plot_models(build_models(dif_hidden=[256, 128], loda_bins=100), X_train3, X_test3, "ex2_6_3d_big_params.png")
    plt.show()

#2.3
# Isolation Forest splits the data with random cuts, and each cut uses only one feature. 
# In 2D this means every cut is a vertical or horizontal line. 
# Because of this, the regions of low scores around the two clusters are not round. 
# They extend in straight vertical and horizontal bands that follow the axes, 
# and points along these bands get lower anomaly scores even when they are far from the data. 
# Areas between the clusters, for example around (10, 10), also get lower scores than the corners, although there is no data there.

#2.5
# LODA builds one-dimensional histograms from random projections of the data, 
# and the score of a point is computed from the probabilities of its projections. 
# A projection gives the same value to all the points on a line perpendicular to the projection direction, 
# so the score map is made of stripes. In our maps the stripes are vertical and horizontal, 
# which suggests that the projections are mostly along the two axes. With 10, 50 and 200 bins the maps look very similar, 
# and only with 3 bins does the map change, showing separate areas instead of the bands. 
# So in this case the number of bins mattered only when it was very small.

#For DIF, the data is first passed through randomly initialised neural networks, 
# and the isolation cuts are made in that new representation. 
# A cut along one axis there does not correspond to a cut along one axis in the original space. 
# In our maps, the scores change gradually and the regions have no straight edges along the axes, unlike with Isolation Forest. 
# Changing the number of neurons ([4, 2], [64, 32] and [256, 128]) made very little difference: the three maps have the same overall pattern, 
# and only the range of the scores changes slightly.