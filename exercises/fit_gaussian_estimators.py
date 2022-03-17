import plotly

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import pyplot as plt

pio.templates.default = "simple_white"
import exercises


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    # raise NotImplementedError()
    new_u_g = UnivariateGaussian()
    array_q1 = np.random.normal(10, 1, size=(1, 1000))
    new_u_g.fit(array_q1)
    print("(" + str(new_u_g.mu_) + "," + str(new_u_g.var_) + ")")

    #####################################
    # Question 2 - Empirically showing sample mean is consistent
    array_q2 = np.linspace(10., 1000., 100)
    # new_u_g.fit(array_q2)
    results = np.array(100)
    for i in range(100):
        new_u_g.fit(array_q2[:(i+1)*10])
        results[i] = abs(new_u_g.mu_-10)
    plt.title("Absolute distance from estimated to actual expectation depend on num of samples")
    plt.xlabel("Sample value")
    plt.ylabel("Absolute distance estimated to actual expectation")
    plt.scatter(array_q2, results)
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model

    new_u_g.fit(array_q1)
    pdf_res = new_u_g.pdf(array_q1)


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0], [0], [4], [0])
    sigma = np.array([1, 0.2, 0, 0.5],
                     [0.2, 2, 0, 0], [0, 0, 1, 0],
                     [0.5, 0, 0, 1])
    new_m_g = MultivariateGaussian()
    array_q4 = np.random.normal(mu, sigma, size=(1, 1000))
    new_m_g.fit(array_q4)
    print(new_m_g.mu_)  # expectation
    print(new_m_g.cov_)

    # Question 5 - Likelihood evaluation
    # f1 =
    # f3 =
    # new_mu = np.array([f1],[0],[f3],[0])
    # new_m_g.log_likelihood(new_mu, new_m_g.cov_, array_q4)

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
