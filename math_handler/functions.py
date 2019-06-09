import numpy as np
from scipy.special import factorial

def poisson(x, lam):
    return (lam ** x / factorial(x)) * np.exp(-lam)


def linear(x, a, b):
    return a + b * x


def poly_3(x, a, b, c, d, e):
    return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4

def prob_poiss(lam_1, lam_2, n):
    x_1 = np.random.poisson(lam_1, n)
    x_2 = np.random.poisson(lam_2, n)
    return int(np.round(np.mean(x_1))), int(np.round(np.mean(x_2)))