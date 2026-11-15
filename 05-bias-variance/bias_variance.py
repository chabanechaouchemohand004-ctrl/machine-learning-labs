"""Lab 5: bias and variance of k-NN, decision trees and random forests; class imbalance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

from common import GREY, NAVY, RED, SEED, load_split, save

N_RUNS = 30
X_train, y_train, X_test, y_test = load_split("biasvar.npz")
y_train, y_test = y_train.astype(int), y_test.astype(int)
print(f"{len(y_train)} training points, {len(y_test)} test points, classes {np.unique(y_train)}, "
      f"class shares {np.round(np.bincount(y_train)[np.unique(y_train)] / len(y_train), 2)}")


def bias_variance(make_model):
    """Train N_RUNS models on random 60 % subsets and compare their test predictions.

    Bias: share of test points whose most frequent prediction is wrong.
    Variance: mean share of runs that disagree with that most frequent prediction.
    """
    preds = np.empty((N_RUNS, len(y_test)), dtype=int)
    for run in range(N_RUNS):
        Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=0.6, random_state=SEED + run, stratify=y_train)
        preds[run] = make_model(run).fit(Xs, ys).predict(X_test)
    votes = np.stack([(preds == c).sum(axis=0) for c in range(preds.max() + 1)])
    main = votes.argmax(axis=0)
    return np.mean(main != y_test), np.mean(1 - votes.max(axis=0) / N_RUNS)


sweeps = {
    "k-NN": (np.arange(1, 50, 2), lambda k: (lambda run: KNeighborsClassifier(n_neighbors=k)), "neighbours $k$"),
    "decision tree": (np.arange(1, 21), lambda d: (lambda run: DecisionTreeClassifier(max_depth=d, random_state=SEED + run)),
                      "maximum depth"),
    "random forest": (np.array([1, 5, 10, 20, 50, 100, 200]),
                      lambda n: (lambda run: RandomForestClassifier(n_estimators=n, random_state=SEED + run, n_jobs=-1)),
                      "number of trees"),
}
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), sharey=True)
for ax, letter, (name, (values, factory, xlabel)) in zip(axes, "abc", sweeps.items()):
    res = np.array([bias_variance(factory(v)) for v in values])
    total = res.sum(axis=1)
    best = total.argmin()
    print(f"[{name}] first: bias {res[0, 0]:.3f}, variance {res[0, 1]:.3f} | last: bias {res[-1, 0]:.3f}, "
          f"variance {res[-1, 1]:.3f} | lowest sum {total[best]:.3f} at {values[best]} "
          f"(bias {res[best, 0]:.3f}, variance {res[best, 1]:.3f})")
    ax.plot(values, res[:, 0], "o-", color=NAVY, label="bias")
    ax.plot(values, res[:, 1], "s--", color=RED, mfc="white", label="variance")
    ax.plot(values, total, "-", color=GREY, lw=1.0, label="bias + variance")
    if name == "random forest":
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_title(f"({letter}) {name}")
axes[0].set_ylabel("share of test points"); axes[0].legend()
fig.tight_layout()
save(fig, __file__, "bias_variance.png")

# ---- Class imbalance ------------------------------------------------------
Xa, ya, Xta, yta = load_split("biasvar_imbalanced.npz")
ya, yta = ya.astype(int), yta.astype(int)
labels = np.unique(yta)
print("imbalanced set, training counts:", dict(zip(*np.unique(ya, return_counts=True))),
      "test counts:", dict(zip(*np.unique(yta, return_counts=True))))


def report(name, model, X, y):
    cm = confusion_matrix(yta, model.fit(X, y).predict(Xta), labels=labels)
    recall = cm.diagonal() / cm.sum(axis=1)
    print(f"[{name}] accuracy {np.trace(cm) / cm.sum():.3f}, recall per class {np.round(recall, 2)}, "
          f"mean recall {recall.mean():.3f}")


largest = np.bincount(ya).max()
parts = [resample(Xa[ya == c], n_samples=largest, replace=True, random_state=SEED) for c in np.unique(ya)]
X_over = np.vstack(parts)
y_over = np.repeat(np.unique(ya), largest)
report("forest, as is", RandomForestClassifier(n_estimators=100, random_state=SEED), Xa, ya)
report("forest, oversampled", RandomForestClassifier(n_estimators=100, random_state=SEED), X_over, y_over)
report("forest, class weights", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED), Xa, ya)
