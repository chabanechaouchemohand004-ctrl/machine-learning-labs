"""Lab 4: decision trees and random forests (depth, number of trees, grid search)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from common import GREY, NAVY, RED, SEED, load_split, save

SHADES = ["#1f3a5f", "#9a3b2e", "#8a8a8a", "#d9d9d9", "#5d7ea8", "#c98b82"]
depths = np.arange(1, 21)


def tree_curve(X_train, y_train, X_test, y_test):
    train, test = [], []
    for depth in depths:
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=SEED).fit(X_train, y_train)
        train.append(tree.score(X_train, y_train)); test.append(tree.score(X_test, y_test))
    return np.array(train), np.array(test)


# ---- Part 1: one tree on a 2-D point set ---------------------------------
X_train, y_train, X_test, y_test = load_split("trees_2d.npz")
classes = np.unique(y_train)
print(f"2-D set: {len(y_train)} training points, {len(y_test)} test points, {len(classes)} classes")

tree3 = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=SEED).fit(X_train, y_train)
cm = confusion_matrix(y_test, tree3.predict(X_test))
tp = cm[0, 0]
precision, recall = tp / cm[:, 0].sum(), tp / cm[0, :].sum()
print(f"depth 3: accuracy {tree3.score(X_test, y_test):.3f}; first class by hand: "
      f"precision {precision:.3f}, recall {recall:.3f}, F1 {2 * precision * recall / (precision + recall):.3f}")

acc_train, acc_test = tree_curve(X_train, y_train, X_test, y_test)
best_depth = depths[acc_test.argmax()]
print(f"best test accuracy {acc_test.max():.3f} at depth {best_depth}; "
      f"depth 20: train {acc_train[-1]:.3f}, test {acc_test[-1]:.3f}")
best_tree = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth, random_state=SEED).fit(X_train, y_train)

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
cmap = ListedColormap(SHADES[:len(classes)])
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 400),
                     np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 400))
for ax, model, title in [(axes[0], tree3, "(a) Regions of a depth-3 tree"),
                         (axes[1], best_tree, f"(b) Regions at depth {best_depth}")]:
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=np.append(classes - 0.5, classes[-1] + 0.5), cmap=cmap, alpha=0.30)
    sub = np.random.default_rng(SEED).choice(len(y_train), 3000, replace=False)
    ax.scatter(X_train[sub, 0], X_train[sub, 1], c=y_train[sub], cmap=cmap, s=1.5, linewidths=0)
    ax.set_title(title); ax.grid(False); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
axes[2].plot(depths, 100 * acc_train, "o-", color=GREY, mfc="white", label="training set")
axes[2].plot(depths, 100 * acc_test, "o-", color=NAVY, label="test set")
axes[2].axvline(best_depth, color=RED, ls="--", lw=0.9)
axes[2].set_xlabel("maximum depth"); axes[2].set_ylabel("accuracy (%)")
axes[2].set_xticks(depths[1::2]); axes[2].set_title("(c) Accuracy against depth"); axes[2].legend()
fig.tight_layout()
save(fig, __file__, "tree_regions_depth.png")

# ---- Part 2: tree against forest on a 100-D set --------------------------
X_train, y_train, X_test, y_test = load_split("trees_100d.npz")
print(f"100-D set: {X_train.shape[0]} training points, {X_test.shape[0]} test points, "
      f"{X_train.shape[1]} features, {len(np.unique(y_train))} classes")
_, tree_test = tree_curve(X_train, y_train, X_test, y_test)
print(f"single tree: best test accuracy {tree_test.max():.3f} at depth {depths[tree_test.argmax()]}")

n_trees = [1, 5, 10, 25, 50, 100, 200, 400, 800]
forest_acc = [RandomForestClassifier(criterion="entropy", n_estimators=n, random_state=SEED, n_jobs=-1)
              .fit(X_train, y_train).score(X_test, y_test) for n in n_trees]
for n, a in zip(n_trees, forest_acc):
    print(f"forest of {n:4d} trees: test accuracy {a:.3f}")

features = [1, 2, 5, 10, 20, 40, 70, 100]
feature_acc = [RandomForestClassifier(criterion="entropy", n_estimators=200, max_features=m, random_state=SEED, n_jobs=-1)
               .fit(X_train, y_train).score(X_test, y_test) for m in features]
for m, a in zip(features, feature_acc):
    print(f"200 trees, {m:3d} features per split: test accuracy {a:.3f}")

grid = GridSearchCV(RandomForestClassifier(criterion="entropy", random_state=SEED),
                    {"n_estimators": [100, 200, 400], "max_depth": [5, 10, None], "max_features": [5, 10, 20]},
                    cv=5, n_jobs=-1).fit(X_train, y_train)
print(f"grid search (5-fold on the training set): {grid.best_params_}, "
      f"cross-validation accuracy {grid.best_score_:.3f}, test accuracy {grid.score(X_test, y_test):.3f}")

fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.1))
a.plot(n_trees, 100 * np.array(forest_acc), "o-", color=NAVY, label="random forest")
a.axhline(100 * tree_test.max(), color=RED, ls="--", lw=0.9, label="best single tree")
a.set_xscale("log"); a.set_xlabel("number of trees"); a.set_ylabel("test accuracy (%)")
a.set_title("(a) Forest size"); a.legend(loc="center right")
b.plot(features, 100 * np.array(feature_acc), "o-", color=NAVY)
b.axvline(10, color=GREY, ls=":", lw=0.9); b.text(11, 100 * min(feature_acc), "$\\sqrt{100}$", color=GREY, fontsize=9)
b.set_xscale("log"); b.set_xlabel("features tried at each split"); b.set_ylabel("test accuracy (%)")
b.set_title("(b) Randomness of the splits, 200 trees")
fig.tight_layout()
save(fig, __file__, "forest_size_features.png")
