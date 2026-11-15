"""Shared plot style and data loaders for the seven labs."""
import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEED = 0

NAVY, RED, GREY, LIGHT, INK = "#1f3a5f", "#9a3b2e", "#6e6e6e", "#c8c8c8", "#222222"
NAVY_MAP = LinearSegmentedColormap.from_list("navy", ["#ffffff", NAVY])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "legend.fontsize": 8.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5,
    "lines.linewidth": 1.3, "lines.markersize": 4, "axes.titlelocation": "left",
    "axes.titlesize": 10})


def save(fig, script_file, name):
    out = Path(script_file).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_faces():
    """LFW subset: 7 people with at least 70 images each (1288 images of 62 x 47 pixels)."""
    local = DATA / "lfw_7people.npy"
    if local.exists():
        images, labels, names = np.load(local, allow_pickle=True)
        return np.asarray(images, dtype=float), np.asarray(labels, dtype=int), list(names)
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(min_faces_per_person=70)
    return lfw.images.astype(float), lfw.target, list(lfw.target_names)


def load_split(name):
    """Course point sets stored as X_train, y_train, X_test, y_test."""
    d = np.load(DATA / name)
    return d["X_train"], d["y_train"].ravel(), d["X_test"], d["y_test"].ravel()


def load_mnist():
    folder = DATA / "mnist"
    if folder.exists():
        def read(file, offset, shape):
            with gzip.open(folder / file) as f:
                return np.frombuffer(f.read(), np.uint8, offset=offset).reshape(shape)
        return ((read("train-images-idx3-ubyte.gz", 16, (-1, 28, 28)), read("train-labels-idx1-ubyte.gz", 8, (-1,))),
                (read("t10k-images-idx3-ubyte.gz", 16, (-1, 28, 28)), read("t10k-labels-idx1-ubyte.gz", 8, (-1,))))
    from tensorflow.keras.datasets.mnist import load_data
    return load_data()


def confusion_plot(ax, cm, labels, title):
    """Row-normalised confusion matrix with the counts written in the cells."""
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap=NAVY_MAP, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7.5,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class"); ax.set_title(title)
