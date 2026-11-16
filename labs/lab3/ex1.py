import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

FIG_DIR = "figures"              # CAN CHANGE: folder where the plots are saved
SEED = 0                       # CAN CHANGE: only changes which random numbers you get
N_BINS = 10                    # CAN CHANGE (used in point 3)
BINS_LIST = [3, 5, 10, 20, 50, 100]   # CAN CHANGE (point 4 just needs different bin counts)
MARGIN = 1.0                   # CAN CHANGE: histogram range = (min - MARGIN, max + MARGIN); must stay wider than the data
N_PROJ = 5                     
N_TRAIN, N_TEST = 500, 500     
TEST_LOW, TEST_HIGH = -3, 3    


def plot_score_map(fig, ax, X_test, scores, title, cmap="viridis", s=12):
    """Scatter the test points coloured by their anomaly score (the 'colormap' the lab asks for)."""
    sc = ax.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap=cmap, s=s)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_title(title)
    fig.colorbar(sc, ax=ax, shrink=0.7)


#1.
def generate_train_data(n_samples=N_TRAIN, seed=SEED):
    """2D standard normal = ONE blob centred at (0,0) with std 1 (make_blobs is required by the lab)."""
    X, _ = make_blobs(n_samples=n_samples, n_features=2,
                      centers=[(0.0, 0.0)], cluster_std=1.0, random_state=seed)
    return X


#2.
def random_unit_vectors(n_proj=N_PROJ, dim=2, seed=SEED):
    """Draw from N(0, I) with np.random.multivariate_normal, then divide by the norm -> unit length."""
    rs = np.random.RandomState(seed)
    W = rs.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=n_proj)
    return W / np.linalg.norm(W, axis=1, keepdims=True)


class SimpleLODA:
    def __init__(self, n_proj=N_PROJ, n_bins=N_BINS, margin=MARGIN, seed=SEED):
        self.n_proj, self.n_bins, self.margin, self.seed = n_proj, n_bins, margin, seed

    def fit(self, X):
        self.W_ = random_unit_vectors(self.n_proj, X.shape[1], self.seed)
        Z = X @ self.W_.T                                   
        self.edges_, self.probs_ = [], []
        for j in range(self.n_proj):
            z = Z[:, j]
            
            counts, edges = np.histogram(z, bins=self.n_bins, range=(z.min() - self.margin, z.max() + self.margin))
            self.edges_.append(edges)
            self.probs_.append(counts / counts.sum())      
        return self

    def score_samples(self, X):
        """Mean over histograms of the probability of the bin each projected sample falls in.
        Samples outside the histogram range get probability 0 for that projection."""
        Z = X @ self.W_.T
        P = np.zeros_like(Z)
        for j in range(self.n_proj):
            idx = np.searchsorted(self.edges_[j], Z[:, j], side="right") - 1
            inside = (idx >= 0) & (idx < self.n_bins)
            P[inside, j] = self.probs_[j][idx[inside]]
        return P.mean(axis=1)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    X_train = generate_train_data()
    model = SimpleLODA().fit(X_train)

    #3.
    X_test = np.random.RandomState(SEED + 1).uniform(low=TEST_LOW, high=TEST_HIGH, size=(N_TEST, 2))
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_score_map(fig, ax, X_test, model.score_samples(X_test), f"Simple LODA, {N_BINS} bins (bright = normal)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "ex1_3_score_map.png"), dpi=120)

    #4.
    fig = plt.figure(figsize=(15, 8))
    for i, nb in enumerate(BINS_LIST, 1):
        m = SimpleLODA(n_bins=nb).fit(X_train)        
        plot_score_map(fig, fig.add_subplot(2, 3, i), X_test, m.score_samples(X_test), f"{nb} bins")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "ex1_4_bins.png"), dpi=120)
    plt.show()


#1.4
# With few bins, each bin covers a wide interval of the projected values,
# so a large area around the centre gets almost the same high score (clearest with 3 bins). 
# As the number of bins increases, the high-score area gets smaller and concentrates around (0, 0), 
# and the score decreases gradually toward the edges. 
# The score values also get smaller with more bins, because each bin contains a smaller part of the training points, so each probability is smaller. 
# The map has straight edges in different directions because a projection only gives information along one direction: 
# all the points on a line perpendicular to the projection vector get the same probability. The final map is the average of five such patterns.