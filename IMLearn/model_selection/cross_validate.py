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
    X_sub_arrays = np.array_split(X, cv)
    y_sub_arrays = np.array_split(y, cv)

    train_score = []
    validation_score = []

    start, end = 0, 0

    for i in range(cv):
        cur_X, cur_y = X, y
        cur_X = np.squeeze(cur_X)
        cur_y = np.squeeze(cur_y)

        start, end = end, end + X_sub_arrays[i].shape[0]
        cur_X = np.delete(cur_X, range(start, end))
        cur_y = np.delete(cur_y, range(start, end))

        model = estimator.fit(cur_X, cur_y)
        validation_x = np.squeeze(X_sub_arrays[i])
        validation_y = np.squeeze(y_sub_arrays[i])

        y_val_pred = model.predict(validation_x)
        all_y_pred = model.predict(cur_X)

        validation_score.append(scoring(validation_y, y_val_pred))
        train_score.append(scoring(cur_y, all_y_pred))
    return np.average(train_score), np.average(validation_score)
