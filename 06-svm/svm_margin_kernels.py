"""Lab 6: support vector machines. Hard margin as a quadratic program, soft margin, kernels."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvxopt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

from common import DATA, GREY, INK, NAVY, RED, save

cvxopt.solvers.options["show_progress"] = False


def hard_margin_primal(X, y):
    """Minimise 0.5 ||w||^2 subject to y_i (w . x_i + b) >= 1, with z = (b, w)."""
    n_points, n_dim = X.shape
    P = np.zeros((n_dim + 1, n_dim + 1)); P[1:, 1:] = np.eye(n_dim)
    q = np.zeros(n_dim + 1)
    G = -y[:, None] * np.hstack([np.ones((n_points, 1)), X])
    h = -np.ones(n_points)
    sol = cvxopt.solvers.qp(*(cvxopt.matrix(m) for m in (P, q, G, h)))
    z = np.array(sol["x"]).ravel()
    return z[1:], z[0], sol["status"]


def draw_points(ax, X, y):
    ax.plot(*X[y == 1].T, "o", color=NAVY, ms=6, label="class +1")
    ax.plot(*X[y == -1].T, "s", color=RED, ms=6, mfc="white", mew=1.3, label="class $-$1")


# ---- Part 1: four points, solved by hand-built QP -------------------------
X = np.array([[0., 0.], [2., 2.], [2., 0.], [3., 0.]])
y = np.array([-1., -1., 1., 1.])
w, b, status = hard_margin_primal(X, y)
margins = y * (X @ w + b)
print(f"QP: w = ({w[0]:.3f}, {w[1]:.3f}), b = {b:.3f}, margin width 2/||w|| = {2 / np.linalg.norm(w):.3f}")
print("y (w.x + b) per point:", np.round(margins, 3), "-> support vectors:", np.where(margins < 1 + 1e-4)[0])
ref = svm.SVC(kernel="linear", C=1e6).fit(X, y)
print(f"scikit-learn, C = 1e6: w = ({ref.coef_[0, 0]:.3f}, {ref.coef_[0, 1]:.3f}), b = {ref.intercept_[0]:.3f}")

# A fifth point of class +1: far from the boundary region, then inside the other class.
cases = {"extra point (2.1, 2.5)": [2.1, 2.5], "extra point (1.5, 2.5)": [1.5, 2.5]}
for name, point in cases.items():
    X5, y5 = np.vstack([X, point]), np.append(y, 1.)
    try:
        w5, b5, status5 = hard_margin_primal(X5, y5)
        feasible = status5 == "optimal" and np.all(y5 * (X5 @ w5 + b5) > 1 - 1e-3)
    except ValueError:      # the solver fails when no line separates the classes
        feasible = False
    line = f"[{name}] hard margin: " + (f"w = ({w5[0]:.2f}, {w5[1]:.2f}), b = {b5:.2f}, "
                                        f"width {2 / np.linalg.norm(w5):.3f}" if feasible else "no separating line")
    print(line)
    for C in [0.1, 1, 10, 100]:
        soft = svm.SVC(kernel="linear", C=C).fit(X5, y5)
        print(f"    soft margin C = {C:5}: width {2 / np.linalg.norm(soft.coef_):.3f}, "
              f"{len(soft.support_)} support vectors, training errors {int(np.sum(soft.predict(X5) != y5))}")

# ---- Part 2: kernels on a 2-D set that is not linearly separable ----------
data = np.load(DATA / "svm_2d.npz", allow_pickle=True)
Xk, yk = data["arr_0"], data["arr_1"].ravel()
print(f"kernel set: {len(yk)} points, classes {np.unique(yk)}")
kernels = {"linear": svm.SVC(kernel="linear", C=1.0), "polynomial, degree 3": svm.SVC(kernel="poly", degree=3, C=1.0),
           "RBF": svm.SVC(kernel="rbf", C=1.0)}
for name, model in kernels.items():
    cv = cross_val_score(model, Xk, yk, cv=5).mean()
    model.fit(Xk, yk)
    print(f"[{name}] training accuracy {model.score(Xk, yk):.3f}, 5-fold accuracy {cv:.3f}, "
          f"{len(model.support_)} support vectors")

fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.0))
ax = axes[0]
xs = np.array([-0.5, 3.6])
for offset, style in [(0, "-"), (1, "--"), (-1, "--")]:
    ax.plot(xs, -(w[0] * xs + b - offset) / w[1], style, color=INK if offset == 0 else GREY, lw=1.0)
draw_points(ax, X, y)
ax.set_xlim(-0.5, 3.6); ax.set_ylim(-1, 3); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("(a) Hard margin from the QP"); ax.legend(loc="upper left")
xx, yy = np.meshgrid(np.linspace(Xk[:, 0].min() - 0.5, Xk[:, 0].max() + 0.5, 300),
                     np.linspace(Xk[:, 1].min() - 0.5, Xk[:, 1].max() + 0.5, 300))
for ax, letter, (name, model) in zip(axes[1:], "bcd", kernels.items()):
    zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-1e9, 0, 1e9], colors=["#f1dedb", "#dde4ee"])
    ax.contour(xx, yy, zz, levels=[0], colors=INK, linewidths=1.0)
    ax.plot(*Xk[yk == yk.max()].T, "o", color=NAVY, ms=2.5)
    ax.plot(*Xk[yk == yk.min()].T, "s", color=RED, ms=2.5, mfc="white", mew=0.8)
    ax.set_title(f"({letter}) {name} kernel"); ax.grid(False); ax.set_xlabel("$x_1$")
fig.tight_layout()
save(fig, __file__, "svm_margin_kernels.png")
