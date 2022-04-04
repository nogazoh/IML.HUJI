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
    new_df = new_df.dropna()
    # new_df = new_df.reset_index()
    new_df.drop(columns=['id'], inplace=True)
    new_df.drop(columns=['date'], inplace=True)
    # new_df = new_df.from_records(new_df, columns=["bedrooms", "bathrooms", "sqft_living",
    #                                               "sqft_lot",
    #                                               "floors", "waterfront", "view", "condition", "grade", "sqft_above",
    #                                               "sqft_basement", "yr_built", "yr_renovated", "zipcode",
    #                                               "sqft_living15",
    #                                               "sqft_lot15", "price"])
    new_df = new_df[new_df['price'] > 0]
    new_df = new_df[new_df['bathrooms'] > 0]
    new_df = new_df[new_df['sqft_lot15'] > 0]
    # todo: add more conditions and explain in file
    return new_df


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = "C:/Users/nogaz/PycharmProjects/IML.HUJI/graphs/ex2") -> NoReturn:
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

    X = X.join(pd.DataFrame({'price': y}))
    covx = X.cov()
    for feature in X.columns:
        fig = plt.figure()
        fig.clear()
        pear = covx[feature]['price'] / (np.std(X[feature]) * np.std(X['price']))
        title = ("Pearson Correlation between "
                 "" + str(feature) + " and price = " + str(pear))
        plt.scatter(X[feature], y)
        plt.xlabel(str(feature))
        plt.ylabel('price')
        plt.title(title)
        plt.savefig(output_path + "/{}.png".format(feature))
        plt.close()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df = load_data(r'C:\Users\nogaz\PycharmProjects\IML.HUJI\datasets\house_prices.csv')
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
    mean_loss = []
    var_loss = []
    model = LinearRegression()
    samp_size = np.linspace(10, 100, num=90)
    for i in range(10, 100):
        f = i / 100
        loss_i = np.ones(10, )
        for j in range(10):
            cur_tr_x = tr_x.sample(frac=f)
            cur_tr_y = tr_y.loc[cur_tr_x.index]
            # cur_tr_y = tr_y.filter(cur_tr_x.index, axis=0)
            model._fit(cur_tr_x.values, cur_tr_y.values)
            cur_loss = model._loss(te_x.values, te_y.values)
            loss_i[j] = cur_loss
        mean_loss.append(np.mean(loss_i))
        var_loss.append(np.std(loss_i))
    fig = go.Figure(
        [go.Scatter(x=samp_size, y=np.array(mean_loss) - (2 * np.array(var_loss)), fill=None, mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=samp_size, y=np.array(mean_loss) + (2 * np.array(var_loss)), fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=samp_size, y=mean_loss, mode="markers+lines", marker=dict(color="black", size=1),
                    showlegend=False)],
        layout=go.Layout(title=r"connection between num of samples and mean of loss of prediction"))
    fig.show()

