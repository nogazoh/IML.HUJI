import plotly

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import math
from matplotlib import pyplot as plt


def test_univariate_gaussian():
    # Question 2 - Empirically showing sample mean is consistent
    arr_si = 1000
    cor_num = 100
    array_q1 = np.random.normal(10, 1, arr_si)
    new_u_g = UnivariateGaussian()
    new_u_g.fit(array_q1)
    print(new_u_g.mu_, new_u_g.var_)
    # Question 2 - Empirically showing sample mean is consistent
    results = np.empty([cor_num, 1])
    array_q2 = np.linspace(10., 1000., num=cor_num)
    for i in range(0, cor_num):
        new_u_g.fit(array_q1[:(i + 1) * 10])
        results[i] = abs(new_u_g.mu_ - 10)
    plt.title("delta between estimated to actual expectation depend on num of samples")
    plt.xlabel("Sample value")
    plt.ylabel("Absolute distance estimated to actual expectation")
    plt.scatter(array_q2, results)
    plt.show()
    # Question 3 - Plotting Empirical PDF of fitted model
    new_u_g.fit(array_q1)
    pdf_res = new_u_g.pdf(array_q1)
    plt.title("Connection between sample val and its PDF val")
    plt.xlabel("Sample value")
    plt.ylabel("PDF vals")
    plt.scatter(array_q1, pdf_res)
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    arr_si = 1000
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0], [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    new_m_g = MultivariateGaussian()
    array_q4 = np.random.multivariate_normal(mu, sigma, arr_si)
    new_m_g.fit(array_q4)
    print(new_m_g.mu_)
    print("\n")
    print(new_m_g.cov_)
    # Question 5 - Likelihood evaluation
    cor_num = 200
    f = np.linspace(-10, 10, cor_num)
    x_values = np.ndarray(cor_num * cor_num)
    y_values = np.ndarray(cor_num * cor_num)
    res_values = np.ndarray(cor_num * cor_num)
    max_q6 = -math.inf
    max_coor_q6 = (0, 0)
    idx = 0
    for i in range(cor_num):
        for j in range(cor_num):
            new_m_g.mu_ = np.array([f[i], 0, f[j], 0])
            cur_val = new_m_g.log_likelihood(new_m_g.mu_, sigma, array_q4)
            if cur_val > max_q6:
                max_q6 = cur_val
                max_coor_q6 = (f[i], f[j])
            res_values[idx] = cur_val
            x_values[idx] = f[i]
            y_values[idx] = f[j]
            idx += 1
    plt.title("Connection elements in mu and likelihood")
    plt.xlabel("first element in mu")
    plt.ylabel("third element in mu")
    plt.scatter(x_values, y_values, None, res_values)
    plt.colorbar()
    plt.show()
    # Question 6 - Maximum likelihood
    print(max_coor_q6)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
