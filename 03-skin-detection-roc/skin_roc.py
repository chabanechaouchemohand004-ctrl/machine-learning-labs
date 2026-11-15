"""Lab 3: skin-pixel detection with a Gaussian likelihood in (Cb, Cr) and ROC analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import multivariate_normal, norm
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

from common import DATA, GREY, NAVY, RED, SEED, save

FOLDER = DATA / "pratheepan"   # image.jpg + ground-truth mask image.png
N_TEST_IMAGES = 4
N_THRESHOLDS = 20


def load_pixels(files):
    """Stack the (Cb, Cr) values and the skin labels of every pixel of the given images."""
    X, y = [], []
    for jpg in files:
        ycbcr = np.array(Image.open(jpg).convert("YCbCr"))
        mask = np.array(Image.open(jpg.with_suffix(".png")))
        X.append(ycbcr[:, :, 1:3].reshape(-1, 2))
        y.append(mask[:, :, 1].reshape(-1) / 255)
    return np.concatenate(X).astype(float), np.concatenate(y).astype(int)


files = sorted(FOLDER.glob("*.jpg"))
X_train, y_train = load_pixels(files[:-N_TEST_IMAGES])
X_test, y_test = load_pixels(files[-N_TEST_IMAGES:])
print(f"{len(files) - N_TEST_IMAGES} training images ({len(y_train)} pixels), "
      f"{N_TEST_IMAGES} test images ({len(y_test)} pixels)")

# One pixel in 1000 keeps the computation short.
X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=1 / 1000, random_state=SEED)
X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=1 / 1000, random_state=SEED)
skin = X_train[y_train == 1]
print(f"kept: {len(y_train)} training pixels ({100 * y_train.mean():.1f} % skin), {len(y_test)} test pixels")

# Model 1: independent Cb and Cr. Model 2: full covariance.
mean, std = skin.mean(axis=0), skin.std(axis=0)
cov = np.cov(skin.T)
print(f"skin mean (Cb, Cr) = ({mean[0]:.1f}, {mean[1]:.1f}), std = ({std[0]:.1f}, {std[1]:.1f}), "
      f"correlation = {cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]):.2f}")
full = multivariate_normal(mean, cov)
models = {
    "independent Cb and Cr": lambda X: norm(mean[0], std[0]).pdf(X[:, 0]) * norm(mean[1], std[1]).pdf(X[:, 1]),
    "full covariance": full.pdf,
}


def rates(p, y, threshold):
    pred = p >= threshold
    tp, tn = np.sum(pred & (y == 1)), np.sum(~pred & (y == 0))
    fp, fn = np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))
    return tp / (tp + fn), tn / (tn + fp), (tp + tn) / len(y)


curves = {}
for name, likelihood in models.items():
    p_train, p_test = likelihood(X_train), likelihood(X_test)
    sens0, spec0, acc0 = rates(p_train, y_train, p_train.mean())
    thresholds = np.linspace(p_train.min(), p_train.max(), N_THRESHOLDS)
    roc = np.array([rates(p_train, y_train, t) for t in thresholds])
    best = np.argmin(np.abs(roc[:, 0] - roc[:, 1]))        # sensitivity closest to specificity
    sens_t, spec_t, acc_t = rates(p_test, y_test, thresholds[best])
    area = auc(1 - roc[:, 1], roc[:, 0])
    curves[name] = (roc, best)
    print(f"[{name}] mean threshold: sens {sens0:.3f}, spec {spec0:.3f}, acc {acc0:.3f}")
    print(f"[{name}] chosen point (train): sens {roc[best, 0]:.3f}, spec {roc[best, 1]:.3f}, "
          f"acc {roc[best, 2]:.3f}; AUC {area:.3f}")
    print(f"[{name}] test: sens {sens_t:.3f}, spec {spec_t:.3f}, acc {acc_t:.3f}")

fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.5))
rng = np.random.default_rng(SEED)
other = X_train[y_train == 0]
other = other[rng.choice(len(other), min(len(other), 2500), replace=False)]
a.plot(other[:, 0], other[:, 1], ".", ms=2, color="#b9b9b9", label="other pixels")
a.plot(skin[:, 0], skin[:, 1], ".", ms=2, color=RED, label="skin pixels")
xx, yy = np.meshgrid(np.linspace(60, 200, 200), np.linspace(90, 210, 200))
d2 = np.einsum("...i,ij,...j", np.dstack([xx, yy]) - mean, np.linalg.inv(cov), np.dstack([xx, yy]) - mean)
a.contour(xx, yy, d2, levels=[1, 4, 9], colors=NAVY, linewidths=0.9)
a.set_xlabel("Cb"); a.set_ylabel("Cr"); a.set_xlim(60, 200); a.set_ylim(90, 210)
a.set_title("(a) Training pixels and fitted Gaussian (1, 2, 3 $\\sigma$)"); a.legend(markerscale=4, loc="upper right")
for (name, (roc, best)), color, marker in zip(curves.items(), [GREY, NAVY], ["s", "o"]):
    b.plot(1 - roc[:, 1], roc[:, 0], marker + "-", color=color, mfc="white" if color == GREY else color, label=name)
    b.plot(1 - roc[best, 1], roc[best, 0], marker, color=RED, ms=7, mfc="none", mew=1.4)
b.plot([0, 1], [0, 1], ":", color="#c8c8c8", lw=0.9)
b.plot([], [], "o", color=RED, mfc="none", mew=1.4, label="chosen threshold")
b.set_xlabel("false-positive rate"); b.set_ylabel("true-positive rate")
b.set_xlim(0, 1); b.set_ylim(0, 1.02); b.set_title("(b) ROC on the training pixels"); b.legend(loc="lower right")
fig.tight_layout()
save(fig, __file__, "skin_scatter_roc.png")
