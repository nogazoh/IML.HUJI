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
    ######
    # q2 part 2
    df_by_m = israel_df.groupby(["Month"], as_index=False).agg("std")
    plt.bar(df_by_m["Month"], df_by_m["Temp"])
    plt.title("Standard Deviation of Temp colored by months")
    plt.ylabel("std")
    plt.xlabel("Month")
    plt.show()
    plt.close()

    # Question 3 - Exploring differences between countries
    df_by_c_m = new_df.groupby(['Country', 'Month'], as_index=False).agg({"Temp": ["mean", "std"]})
    px.line(x=df_by_c_m["Month"], y=df_by_c_m.iloc[:, 2], error_y=df_by_c_m.iloc[:, 3],
            color=df_by_c_m.iloc[:, 0], labels=dict(x="Month", y="Temp"),
            title="Average Temperature per Month").show()

    # Question 4 - Fitting model for different values of `k`
    temp_isr = israel_df['Temp']
    israel_df = israel_df.drop('Temp', axis=1)
    train_x, train_y, test_x, test_y = split_train_test(israel_df["DayOfYear"], temp_isr)
    loss_k = np.ones(10, )
    for k in range(1, 11):
        cur_pol = PolynomialFitting(k)
        cur_pol._fit(train_x, train_y)
        to_print = round(cur_pol._loss(test_x.values, test_y.values), 2)
        print(to_print)
        loss_k[k - 1] = to_print
    plt.bar(range(1,11), loss_k)
    plt.title("loss per polynomial degree")
    plt.ylabel("loss")
    plt.xlabel("polynomial degree")
    plt.show()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
