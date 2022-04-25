import matplotlib.pyplot as plt
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", r'C:\Users\nogaz\PycharmProjects\IML.HUJI\datasets\linearly_separable.npy'),
                 (
                         "Linearly Inseparable",
                         r'C:\Users\nogaz\PycharmProjects\IML.HUJI\datasets\linearly_inseparable.npy')]:
        # Load dataset
        X, y = load_dataset(f)
        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback_helper(fit: Perceptron, x: np.ndarray, y_: int):
            losses.append(fit._loss(X, y))

        perceptron = Perceptron(callback=callback_helper)
        perceptron.fit(X, y)

        # Plot figure
        ln = len(losses)
        sump = list(range(1, ln + 1))
        plt.plot(sump, losses)
        plt.title(n)
        plt.ylabel("losses")
        plt.xlabel("iteration num")
        plt.show()
        plt.close()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")



def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in [r'C:\Users\nogaz\PycharmProjects\IML.HUJI\datasets\gaussian1.npy',
              r'C:\Users\nogaz\PycharmProjects\IML.HUJI\datasets\gaussian2.npy']:
        # Load dataset
        X, y = load_dataset(f)
        # Fit models and predict over training set
        lda_new = LDA()
        lda_new.fit(X, y)
        y_pred_lda = lda_new.predict(X)
        ga_na_ba = GaussianNaiveBayes()
        ga_na_ba.fit(X,y)
        y_pred_gnb = ga_na_ba.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy

        l_accuracy = accuracy(y, y_pred_lda)
        g_accuracy = accuracy(y, y_pred_gnb)
        symbols = np.array(["circle", "diamond", "square"])
        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"Gaussian Naive Bayes, accuracy: {g_accuracy}",
                                                            rf"Linear Discriminant Analysis, accuracy: {l_accuracy}"],
                            horizontal_spacing=0.01, vertical_spacing=.03)

        # Add traces for data-points setting symbols and colors
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=y_pred_gnb, symbol=symbols[y])),
                      1, 1)
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=y_pred_lda, symbol=symbols[y])),
                      1, 2)
        fig.show()
        # Add `X` dots specifying fitted Gaussians' means


        # Add ellipses depicting the covariances of the fitted Gaussians



if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
