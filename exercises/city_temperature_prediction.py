import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
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
    # days = np.linspace(1.0, 365.0, 10)
    file = pd.read_csv(filename, parse_dates=True)  # todo check
    df = pd.DataFrame(file)
    df = df.reset_index()
    df['DayOfYear'] = pd.to_datetime(df[['Year', 'Month', 'Day']])  # todo check
    df = df.from_records(df, columns=["Country", "City", "Date", "Year", "Month", "day","DayOfYear", "Temp"])
    df = df[df['Temp'] > -25]
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    new_df = load_data('../datasets/City_Temperature.csv')
    temp = new_df['Temp']
    df = new_df.drop('Temp', axis=1)
    # Question 2 - Exploring data for specific country
    israel_df = df[df["Country"] == "Israel"]
    isr_temp = np.ndarray(len(israel_df),)
    dayodyear = np.ndarray(len(israel_df), )
    color = np.ndarray(len(israel_df), )
    idx = 0
    for row in israel_df.itertuples():
        isr_temp[idx] = (row["Temp"])
        dayodyear[idx] = row["DayOfYear"]
        color[idx] = row["Month"]
        print(row.Index)
        idx += 1
    plt.title("Temp in Israel as a function of Day of year")
    plt.xlabel("Day of year")
    plt.ylabel("temp in Israel")
    plt.scatter(dayodyear, isr_temp)
    plt.show()
    raise NotImplementedError()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
