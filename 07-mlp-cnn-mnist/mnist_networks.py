"""Lab 7: from logistic regression to a small CNN on MNIST (TensorFlow 2 / Keras)."""
import os
import sys
import time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from common import GREY, NAVY, RED, SEED, confusion_plot, load_mnist, save

EPOCHS = 20
tf.keras.utils.set_random_seed(SEED)

# One tenth of MNIST, as in the lab: 6000 training and 1000 test images.
(X_train, y_train), (X_test, y_test) = load_mnist()
X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=SEED, stratify=y_train)
X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=0.1, random_state=SEED, stratify=y_test)
X_train = X_train[..., None].astype("float32") / 255
X_test = X_test[..., None].astype("float32") / 255
Y_train = tf.keras.utils.to_categorical(y_train, 10)
Y_test = tf.keras.utils.to_categorical(y_test, 10)
print(f"train {X_train.shape}, test {X_test.shape}")


def build(layers):
    inputs = Input(shape=(28, 28, 1))
    x = inputs
    for layer in layers:
        x = layer(x)
    return Model(inputs, Dense(10, activation="softmax")(x))


def sgd(lr, momentum=0.0):
    return lambda: tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)


def adam():
    return tf.keras.optimizers.Adam(learning_rate=1e-3)


conv = lambda: [Conv2D(32, (3, 3), activation="relu"), Conv2D(64, (3, 3), activation="relu")]
networks = {
    "logistic regression": (lambda: [Flatten()], sgd(0.1), 256),
    "MLP, 256 units": (lambda: [Flatten(), Dense(256, activation="relu")], sgd(0.01, 0.9), 128),
    "MLP, 256 units, dropout 0.5": (lambda: [Flatten(), Dense(256, activation="relu"), Dropout(0.5)], sgd(0.01, 0.9), 128),
    "MLP, 256 + 128 units, dropout 0.3": (lambda: [Flatten(), Dense(256, activation="relu"), Dropout(0.3),
                                                   Dense(128, activation="relu"), Dropout(0.3)], adam, 128),
    "CNN, 2 conv layers": (lambda: conv() + [Flatten(), Dense(256, activation="relu")], adam, 128),
    "CNN, max-pooling, dropout": (lambda: conv() + [MaxPooling2D((2, 2)), Dropout(0.25), Flatten(),
                                                    Dense(256, activation="relu"), Dropout(0.5)], adam, 128),
}

results = {}
for name, (layers, optimizer, batch) in networks.items():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = build(layers())
    model.compile(loss="categorical_crossentropy", optimizer=optimizer(), metrics=["accuracy"])
    start = time.perf_counter()
    history = model.fit(X_train, Y_train, batch_size=batch, epochs=EPOCHS, verbose=0,
                        validation_data=(X_test, Y_test)).history
    elapsed = time.perf_counter() - start
    pred = model.predict(X_test, verbose=0).argmax(axis=1)
    results[name] = dict(params=model.count_params(), history=history, pred=pred, time=elapsed)
    print(f"[{name}] parameters {model.count_params():,}, train accuracy {history['accuracy'][-1]:.3f}, "
          f"test accuracy {history['val_accuracy'][-1]:.3f}, best epoch {max(history['val_accuracy']):.3f}, "
          f"{elapsed:.0f} s")

# Parameter counts by hand.
print("by hand: logistic", 784 * 10 + 10, "| MLP", 784 * 256 + 256 + 256 * 10 + 10,
      "| CNN", (3 * 3 * 1 * 32 + 32) + (3 * 3 * 32 * 64 + 64) + (24 * 24 * 64 * 256 + 256) + (256 * 10 + 10),
      "| CNN with pooling", (3 * 3 * 1 * 32 + 32) + (3 * 3 * 32 * 64 + 64) + (12 * 12 * 64 * 256 + 256) + (256 * 10 + 10))

styles = [(GREY, ":"), (GREY, "--"), (GREY, "-"), (RED, "--"), (NAVY, "--"), (NAVY, "-")]
fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
epochs = np.arange(1, EPOCHS + 1)
for (name, res), (color, style) in zip(results.items(), styles):
    a.plot(epochs, 100 * np.array(res["history"]["val_accuracy"]), style, color=color, label=name)
a.set_ylim(84, 100); a.set_xticks(epochs[1::2]); a.set_xlabel("epoch"); a.set_ylabel("test accuracy (%)")
a.set_title("(a) Test accuracy during training")
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
last = list(results)[-1]
confusion_plot(b, confusion_matrix(y_test, results[last]["pred"]), list(range(10)), "(b) Confusion matrix, CNN with pooling")
b.set_xticklabels(range(10), rotation=0, ha="center")
fig.tight_layout()
save(fig, __file__, "mnist_accuracy_confusion.png")
