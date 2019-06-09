import matplotlib.pyplot as pl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List
import numpy as np
from scipy.optimize import curve_fit
from math_handler import poisson,poly_3

def scipyFit(x, y, method,p0 = None,boundaries = (-np.inf, np.inf),sigma = None):
    if boundaries is not None and len(boundaries) != 2:
        raise ValueError("Boundaries need to be a two 2D tuple")

    if p0 is not None and boundaries is not None and boundaries != (-np.inf, np.inf) and len(p0) != len(boundaries[0]) :
        raise ValueError("P0 and Fixed Array have to have the same length")

    popt, pcov = curve_fit(method, x, y,p0=p0,bounds = boundaries,sigma=sigma)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def plot_poisson(vals,num,popt,median_p,ax : Axes):
    x_plot = np.linspace(0, max(vals), num=500)
    ax.bar(vals,num,alpha=0.5,label='Goal distribution')
    ax.set_title('Median win probability: '+'%.2f' % median_p)
    ax.plot(x_plot, poisson(x_plot, *popt), color='red', label='fit')
    ax.legend()

def plot_lambda(arr_l,popt_poly):
    x_plot = np.linspace(min(arr_l[0]), max(arr_l[0]), num=500)
    pl.figure(figsize=(16, 10))
    pl.errorbar(arr_l[0], arr_l[1], yerr=arr_l[2], fmt='o')
    pl.plot(x_plot, poly_3(x_plot, *popt_poly))
    pl.title(r"$\lambda$ value distribution")

def create_poisson_distributions(p_list,plot=False):
    if plot:
        fig, ax_list = pl.subplots(2, 4, figsize=(16, 10), sharex=True)
        fig: Figure
        ax_list: List[Axes]

    p_range = np.array((np.arange(0.11, 0.91, 0.1), np.arange(0.21, 1.01, 0.1))).T
    lam_list = []
    max_goals = np.amax(np.unique(p_list[1]))

    for i, (p_l, p_u) in enumerate(p_range):
        mask = np.logical_and(p_list[0] > p_l, p_list[0] < p_u)
        vals, num = np.unique(p_list[1][mask], return_counts=True)

        while len(vals) <= max_goals:
            vals = np.array(vals.tolist() + [max(vals) + 1])
            num = np.array(num.tolist() + [0])

        num = num / np.sum(num)
        median_p = np.median(p_list[0][mask])

        popt,perr = scipyFit(vals,num,poisson)

        x = 0 if i < 4 else 1
        y = i if i < 4 else i - 4

        if plot:
            plot_poisson(vals,num,popt,median_p,ax_list[x,y])

        lam_list.append((median_p, popt[0], perr[0]))

    arr_l = np.array(lam_list).T
    popt_poly,perr_poly = scipyFit(arr_l[0],arr_l[1],poly_3,sigma=arr_l[2])

    if plot:
        plot_lambda(arr_l,popt_poly)
        pl.show()

    return popt_poly