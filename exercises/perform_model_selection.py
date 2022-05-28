from __future__ import annotations
import numpy as np
import pandas as pd
import plotly
from pandas import DataFrame
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    all_X = np.linspace(-1.2, 2.0, n_samples)
    scale_wo_noise = np.vectorize(lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2))
    all_y_wo_noise = scale_wo_noise(all_X)
    all_y_noise = all_y_wo_noise + np.random.normal(size=len(all_y_wo_noise), scale=noise)
    test_portion = 2 / 3

    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(all_X), pd.Series(all_y_noise), test_portion)
    fig = go.Figure()
    fig.add_traces([go.Scatter(x=all_X, y=all_y_wo_noise, mode="markers", name="All noiseless samples",
                               marker=dict(color="blue", opacity=.7), showlegend=True),
                    go.Scatter(x=train_x.to_numpy()[:, 0], y=train_y.to_numpy(), mode="markers",
                               marker=dict(color="red", opacity=.7), name="Train noisy samples",
                               showlegend=True),
                    go.Scatter(x=test_x.to_numpy()[:, 0], y=test_y.to_numpy(), mode="markers",
                               marker=dict(color="green", opacity=.7), name="Test noisy samples",
                               showlegend=True)
                    ]).update_layout(title="Samples display",
                       xaxis_title="X vals",
                       yaxis_title="y vals" )\
        # .show()
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    av_train_err = []
    av_validation_err = []
    for k in range(0, 11):
        cur_model = PolynomialFitting(k)
        t_s, v_s = cross_validate(PolynomialFitting(k), train_x, train_y,
                                  mean_square_error)  # todo check
        av_train_err.append(t_s)
        av_validation_err.append(v_s)
    k_x = list(range(11))
    fig = go.Figure()
    fig.add_traces([go.Scatter(x=k_x, y=av_train_err, mode="markers+lines", name="Average training err",
                               marker_color = 'rgb(152,171,150)'),
                    go.Scatter(x=k_x, y=av_validation_err, mode="markers+lines", name="Average validation err",
                               marker_color = 'rgb(25,115,132)'), #25,115,132
                    ]).update_layout(title="Average training and validation err",
                       xaxis_title="polynomial degree",
                       yaxis_title="average err" )\
        # .show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_k = av_validation_err.index(min(av_validation_err))
    # train_x = train_x.to_numpy()
    # train_y = train_y.to_numpy()
    # cur_model = PolynomialFitting(min_k).fit(np.squeeze(train_x), np.squeeze(train_y))
    # print(min_k, mean_square_error(np.squeeze(test_y), round(cur_model.predict(test_x), 2))) #todo fix


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x = np.delete(X, range(n_samples, X.shape[0]))
    test_x = np.delete(X, range(0, n_samples))
    train_y = np.delete(y, range(n_samples, X.shape[0]))
    test_y = np.delete(y, range(0, n_samples))

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    # cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
    # scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]



    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
