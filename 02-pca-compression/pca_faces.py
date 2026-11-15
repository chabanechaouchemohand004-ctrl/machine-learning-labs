"""Lab 2: principal component analysis for classification and compression (LFW, 7 people)."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from common import NAVY, RED, SEED, load_faces, save

images, labels, names = load_faces()
X = images.reshape(len(images), -1)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=SEED)

# Full decomposition: how many components carry the variance.
full = PCA(random_state=SEED).fit(X_train)
cumulative = np.cumsum(full.explained_variance_ratio_)
n95 = int(np.argmax(cumulative >= 0.95)) + 1
print(f"dimension {X.shape[1]}, components for 95 % of the variance: {n95}")


def knn_score(train, test):
    start = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=5, metric="manhattan").fit(train, y_train)
    score = accuracy_score(y_test, knn.predict(test))
    return score, time.perf_counter() - start


# Classification: 5-NN on the pixels, then on 100 principal components.
pca100 = PCA(n_components=100, random_state=SEED).fit(X_train)
acc_raw, t_raw = knn_score(X_train, X_test)
acc_pca, t_pca = knn_score(pca100.transform(X_train), pca100.transform(X_test))
print(f"5-NN on {X.shape[1]} pixels: accuracy {acc_raw:.3f}, {1e3 * t_raw:.0f} ms")
print(f"5-NN on 100 components:  accuracy {acc_pca:.3f}, {1e3 * t_pca:.0f} ms")

# Compression: mean Euclidean reconstruction error on the test images.
counts = [10, 25, 50, 100, 200, 400]
errors, kept = [], []
for n in counts:
    pca = PCA(n_components=n, random_state=SEED).fit(X_train)
    rebuilt = pca.inverse_transform(pca.transform(X_test))
    errors.append(np.mean(np.linalg.norm(rebuilt - X_test, axis=1)))
    kept.append(pca.explained_variance_ratio_.sum())
    print(f"{n:4d} components: variance kept {100 * kept[-1]:.1f} %, error {errors[-1]:.1f}, "
          f"size divided by {X.shape[1] / n:.0f}")
norm_ref = np.mean(np.linalg.norm(X_test - X_train.mean(axis=0), axis=1))
print(f"error when every image is replaced by the mean face: {norm_ref:.1f}")

fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.1))
a.plot(np.arange(1, len(cumulative) + 1), 100 * cumulative, color=NAVY)
a.axhline(95, color=RED, ls="--", lw=0.9)
a.axvline(n95, color=RED, ls="--", lw=0.9)
a.text(n95 * 0.85, 40, f"95 % with\n{n95} components", color=RED, fontsize=9, ha="right")
a.set_xscale("log"); a.set_xlabel("number of components"); a.set_ylabel("cumulative variance (%)")
a.set_title("(a) Variance kept")
b.plot(counts, errors, "o-", color=NAVY)
b.set_xscale("log"); b.set_xticks(counts); b.set_xticklabels(counts); b.minorticks_off()
b.set_xlabel("number of components"); b.set_ylabel("mean reconstruction error")
b.set_title("(b) Reconstruction error, test images")
fig.tight_layout()
save(fig, __file__, "pca_variance_reconstruction.png")
