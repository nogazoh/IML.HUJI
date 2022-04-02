import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from matplotlib import pyplot as plt

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    file = pd.read_csv(filename)
    new_df = pd.DataFrame(file)
    new_df = new_df.reset_index()
    column_names = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']
    new_df['sqft_Total'] = new_df[column_names].sum(axis=1)
    new_df = new_df.from_records(new_df, columns=["id", 'sqft_Total', "date", "bedrooms", "bathrooms", "sqft_living",
                                                  "sqft_lot",
                                                  "floors", "waterfront", "view", "condition", "grade", "sqft_above",
                                                  "sqft_basement", "yr_built", "yr_renovated", "zipcode",
                                                  "sqft_living15",
                                                  "sqft_lot15", "price"])
    new_df = new_df[new_df['price'] > 0]
    new_df = new_df[new_df['id'] > 0]
    new_df = new_df[new_df['bathrooms'] > 0]
    #todo: add more conditions and explain in file
    return new_df


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    stdy = np.std(y)
    for feature in X:
        new_cov = np.cov(df[feature].values, y)
        stdx = np.std(df[feature].values)
        pc = new_cov / stdy * stdx
        plt.title("Pearson Correlation between "
                  "" + str(feature) + " and price = " + str(pc))
        plt.xlabel(str(feature))
        plt.ylabel("price")
        plt.scatter(df[feature].values, y)
        plt.savefig(output_path + feature)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df = load_data('../datasets/house_prices.csv')
    price = df['price']
    df = df.drop('price', axis=1)
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, price)

    # Question 3 - Split samples into training- and testing sets.
    tr_x, tr_y, te_x, te_y = split_train_test(df, price)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    mean_loss = np.ones(90, )
    var_loss = np.ones(90, )
    for i in range(10, 101):
        cur_tr_x, cur_tr_y, cur_te_x, cur_te_y = split_train_test(tr_x, tr_y, float(1 / i))
        loss_i = np.ones(10, )
        for j in range(10):
            model = LinearRegression(True)
            model.fit(cur_tr_x.to_numpy(), cur_tr_y.to_numpy())
            model.predict(te_x.to_numpy())
            cur_loss = model.loss(te_x.to_numpy(), te_y.to_numpy())
            loss_i[j] = cur_loss
        mean_loss[i] = np.mean(loss_i)
        var_loss[i] = np.var(loss_i)
        plt.close()
    prec = np.linspace(10.0, 100.0, 10)
    plt.title("connection between num of samples and mean of loss of prediction")
    plt.xlabel("percentage")
    plt.ylabel("mean loss")
    plt.scatter(prec, mean_loss)
    plt.show()
    #todo # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

