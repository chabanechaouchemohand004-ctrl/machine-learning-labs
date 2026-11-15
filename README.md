# Machine learning labs: from k-NN to a small CNN

Seven labs in Python on classification, dimension reduction, model evaluation and neural networks. Master Systèmes Communicants (M1), Sorbonne Université. Every number below comes from the scripts in this repository.

| Folder | What I did | Tools |
| --- | --- | --- |
| [`01-knn-face-recognition/`](01-knn-face-recognition/README.md) | Face recognition on raw pixels with k-NN, confusion matrix, choice of k | scikit-learn |
| [`02-pca-compression/`](02-pca-compression/README.md) | PCA on face images: variance kept, k-NN on 100 components, reconstruction error | scikit-learn |
| [`03-skin-detection-roc/`](03-skin-detection-roc/README.md) | Skin-pixel detector from a Gaussian likelihood in (Cb, Cr), ROC curve, operating point | NumPy, SciPy, Pillow |
| [`04-decision-tree-random-forest/`](04-decision-tree-random-forest/README.md) | Tree depth and overfitting, random forest against a single tree, grid search | scikit-learn |
| [`05-bias-variance/`](05-bias-variance/README.md) | Bias and variance of k-NN, trees and forests over 30 trainings; class imbalance | scikit-learn |
| [`06-svm/`](06-svm/README.md) | Hard-margin SVM solved as a quadratic program, soft margin, kernels | cvxopt, scikit-learn |
| [`07-mlp-cnn-mnist/`](07-mlp-cnn-mnist/README.md) | Logistic regression, MLP and CNN on MNIST, dropout and max-pooling | TensorFlow 2 / Keras |

## Results at a glance

| Lab | Result |
| --- | --- |
| 1 | 7-person face recognition: **63.0 %** with 4-NN on raw pixels, against 41.0 % for the largest class |
| 2 | 100 principal components out of 2914 pixels: **64.3 %** against 62.4 % on the pixels, prediction about 35 times shorter |
| 3 | Full-covariance Gaussian: **82.0 %** of test pixels classified, AUC 0.859 against 0.831 with independent Cb and Cr |
| 4 | 100-D, 10 classes: forest of 800 trees **47.7 %**, best single tree 31.3 % |
| 5 | k-NN error split: lowest bias + variance at k = 9 (0.051 + 0.018) |
| 6 | Hard-margin QP and scikit-learn give the same line: w = (1, −1), b = −1 |
| 7 | 6000 MNIST images: logistic regression 88.9 %, MLP 95.7 %, CNN **97.3 %** on 1000 test images |

## Tools

| Tool | Use |
| --- | --- |
| scikit-learn | k-NN, PCA, trees, forests, SVM, grid search, metrics |
| TensorFlow 2 / Keras | Dense and convolutional networks |
| cvxopt | Quadratic program of the hard-margin SVM |
| NumPy, SciPy, Pillow | Arrays, Gaussian densities, image reading |
| Matplotlib | Figures, one shared style in `common.py` |

## Run

```bash
pip install -r requirements.txt
python 01-knn-face-recognition/knn_faces.py
python 02-pca-compression/pca_faces.py
python 03-skin-detection-roc/skin_roc.py
python 04-decision-tree-random-forest/trees_forest.py
python 05-bias-variance/bias_variance.py
python 06-svm/svm_margin_kernels.py
python 07-mlp-cnn-mnist/mnist_networks.py
```

Labs 1, 2 and 7 download their public data sets (LFW through scikit-learn, MNIST through Keras). Labs 3 to 6 read their data from a local `data/` folder: the Pratheepan skin images for lab 3 and the course point sets for labs 4 to 6. The random seed is fixed in `common.py`.

## Credits

The exercises and the data sets come from the machine learning course. The scripts, the analysis and the figures are mine. Course statements, reports and course data files are not included.
