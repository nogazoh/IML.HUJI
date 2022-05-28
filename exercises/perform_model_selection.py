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
    def F_x(x):
        return (x + 3) * (x + 2) * (x - 1) * (x - 2)

    all_X = np.linspace(-1.2, 2, n_samples)
    # scale_noise = lambda x: F_x(x, noise)
    scale_wo_noise = lambda x: F_x(x)
    all_y_wo_noise = scale_wo_noise(all_X)
    all_y_noise = all_y_wo_noise + np.random.normal(size=len(all_y_wo_noise), scale=noise)
    test_portion = 2 / 3
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(all_X), pd.Series(all_y_noise), test_portion)

    go.Figure([go.Scatter(name='All noiseless samples', x=all_X, y=all_y_wo_noise, mode='markers',
                          marker_color='rgb(152,171,150)'),
               go.Scatter(name='Train noisy samples', x=train_x.to_numpy(), y=test_y.to_numpy(),
                          # todo fix - it is not showing
                          mode='markers',
                          marker_color='rgb(25,115,132)'),  # todo fix - it is not showing
               go.Scatter(name='Test noisy samples', x=test_x.to_numpy(), y=test_y.to_numpy(),
                          mode='markers',
                          marker_color='rgb(204, 110, 188)')]) \
        .update_layout(title="Samples display",
                       xaxis_title="X vals",
                       yaxis_title="y vals").show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_x = train_x.to_numpy()# todo why isnt it working
    train_y = train_y.to_numpy()  # todo why isnt it working
    test_x = test_x.to_numpy()# todo why isnt it working
    test_y = test_y.to_numpy()  # todo why isnt it working
    av_train_err = []
    av_validation_err = []
    for k in range(0, 11):
        cur_model = PolynomialFitting(k).fit(train_x, train_y)
        t_s, v_s = cross_validate(cur_model, train_x, train_y,
                                  cur_model.loss(test_x, test_y))  # todo check
        av_train_err.append(t_s)
        av_validation_err.append(v_s)
    k_x = [range(11)]
    go.Figure([go.Scatter(name='Average training err', x=k_x, y=av_train_err, mode='markers',
                          marker_color='rgb(152,171,150)'),
               go.Scatter(name='Average validation err', x=k_x, y=av_validation_err,  # todo fix - it is not showing
                          mode='markers',
                          marker_color='rgb(25,115,132)'),
               ]) \
        .update_layout(title="Average training and validation err",
                       xaxis_title="polynomial degree",
                       yaxis_title="average err").show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_k = av_validation_err.index(min(av_validation_err))
    cur_model = PolynomialFitting(min_k).fit(train_x, train_y)
    print(min_k, mean_square_error(test_y, round(cur_model.predict(test_x), 2)))


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



    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
