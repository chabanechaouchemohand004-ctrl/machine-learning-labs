"""Lab 1: k-nearest-neighbour face recognition on raw pixels (LFW, 7 people)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from common import NAVY, RED, SEED, confusion_plot, load_faces, save

images, labels, names = load_faces()
X = images.reshape(len(images), -1)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=SEED)

# The scaler is fitted on the training images only, then applied to both sets.
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

ks = np.arange(1, 21)
acc = []
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean").fit(X_train, y_train)
    acc.append(accuracy_score(y_test, knn.predict(X_test)))
acc = np.array(acc)
best_k = ks[acc.argmax()]

knn1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
cm = confusion_matrix(y_test, knn1.predict(X_test))
majority = np.bincount(y_test).max() / len(y_test)

print(f"images {images.shape}, classes {len(names)}, train {len(y_train)}, test {len(y_test)}")
print("images per class:", dict(zip(names, np.bincount(labels))))
print(f"1-NN accuracy {acc[0]:.3f} (from the matrix: {np.trace(cm) / cm.sum():.3f})")
print(f"best k = {best_k}, accuracy {acc.max():.3f}; always-majority baseline {majority:.3f}")
recall = cm.diagonal() / cm.sum(axis=1)
print("1-NN recall per class:", {n: round(r, 2) for n, r in zip(names, recall)})

fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.5), gridspec_kw={"width_ratios": [1, 1.05]})
a.plot(ks, 100 * acc, "o-", color=NAVY, label="k-NN, test split")
a.axhline(100 * majority, color=RED, ls="--", lw=0.9, label="always the largest class")
a.set_xlabel("number of neighbours $k$"); a.set_ylabel("accuracy (%)")
a.set_xticks(ks[::2]); a.set_title("(a) Accuracy against $k$"); a.legend()
short = [n.split()[-1] for n in names]
confusion_plot(b, cm, short, "(b) Confusion matrix, 1-NN")
fig.tight_layout()
save(fig, __file__, "knn_accuracy_confusion.png")
