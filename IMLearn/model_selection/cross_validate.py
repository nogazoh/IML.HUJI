from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds. יעני לכמה אני מחלקת את זה

    Returns
    -------
    train_score: float
        Average train score over folds // loss during training

    validation_score: float
        Average validation score over folds
    """
    X_sub_arrays = np.split(X, cv)
    y_sub_arrays = np.split(y, cv)
    wo_folds = []
    folds = []
    for i in range(cv):
        cur_X, cur_y = X, y
        cur_X = np.delete(cur_X, X_sub_arrays[i])
        cur_y = np.delete(cur_y, y_sub_arrays[i])
        model = estimator.fit(cur_X, cur_y)
        y_pred = model.predict(X)
        folds.append(scoring(X_sub_arrays[i], y_sub_arrays[i], cv))
        wo_folds.append(scoring(cur_X, y_pred, cv))
    return np.average(wo_folds), np.average(folds)
