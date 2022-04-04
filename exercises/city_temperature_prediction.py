from boto import sns

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from matplotlib import pyplot as plt

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    tmp_df = pd.read_csv(filename, parse_dates=["Date"])
    tmp_df = tmp_df.reset_index()
    tmp_df = tmp_df.dropna()
    tmp_df['DayOfYear'] = tmp_df["Date"].dt.dayofyear
    # tmp_df = tmp_df.from_records(tmp_df, columns=["Country", "City",
    #                                               "Year", "Month", "Day", "DayOfYear", "Temp"])
    tmp_df = tmp_df[tmp_df['Temp'] > -25]
    return tmp_df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    new_df = load_data(r'C:\Users\nogaz\PycharmProjects\IML.HUJI\datasets\City_Temperature.csv')
    # temp = new_df['Temp']
    # df = new_df.drop('Temp', axis=1)
    # Question 2 - Exploring data for specific country
    israel_df = new_df[new_df["Country"] == "Israel"]
    israel_df = israel_df[israel_df["Temp"] > -10]
    israel_df["Year_c"] = israel_df["Year"].astype(str)
    colors = israel_df["Year"].unique()
    rgb_values = sns.color_palette("Set2", colors.shape[0])
    c_per_m = dict(zip(colors, rgb_values))
    plt.scatter(israel_df["DayOfYear"], israel_df["Temp"], c=israel_df["Year"].map(c_per_m))
    plt.xlabel("Day Of Year")
    plt.ylabel("Temp")
    plt.title("Temperature per Day")
    plt.show()
    # plt.title("Temp in Israel as a function of Day of year")
    # plt.xlabel("Day of year")
    # plt.ylabel("temp in Israel")
    # plt.show()
    ######
    # q2 part 2
    df_by_m = new_df.groupby(["Month"]).agg({"temp": ["std"]})

    # Question 3 - Exploring differences between countries
    df_by_c_m = new_df.groupby(['Country', 'Month']).agg({"temp": ["mean", "std"]})

    new_size = df_by_c_m.size

    # raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    temp_isr = israel_df['Temp']
    israel_df = israel_df.drop('Temp', axis=1)
    train_x, train_y, test_x, test_y = split_train_test(israel_df, temp_isr)
    loss_k = np.ones(10, )
    for k in range(1, 11):
        cur_pol = PolynomialFitting(k)
        cur_pol.fit(train_x.values, train_y.values)
        loss_k[k - 1] = round(cur_pol.loss(test_x.values, test_y.values), 2)

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
