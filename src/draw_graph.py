import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import cal_pdf
from src.Lognormal_mixture_model import Lognormal_mixture_model_curve

plt.style.use('ggplot')


def draw_mixed_process(r, lambdas, sigmas, n=500):
    dw = np.random.randn(n)
    ss_ls = []
    for sigma in sigmas:
        ds_s = 0.01 * r + sigma * dw * 0.1
        s = np.ones((n + 1,))
        for step in range(n):
            s[step + 1] = ds_s[step] * s[step] + s[step]
        ss_ls.append(s)
    ss = np.vstack(ss_ls)
    ss_total = np.sum(lambdas.reshape((-1, 1)) * ss, axis=0)
    fig, ax = plt.subplots()
    for s in ss_ls:
        ax.plot(s, ls='--')
    ax.plot(ss_total, color='g')
    ax.axhline(1, ls='-.', color='grey')
    plt.xticks([])
    plt.show()


def draw_multimeans_process(r, lambdas, xis, sigmas, n=500):
    dw = np.random.randn(n)
    ss_ls_1 = []
    ss_ls_2 = []
    for i in range(4):
        ds_s_1 = 1 / n * r + sigmas[i] * dw * np.sqrt(1 / n)
        ds_s_2 = 1 / n * (np.log(xis[i]) + r) + sigmas[i] * dw * np.sqrt(1 / n)
        s_1 = np.ones((n + 1,))
        s_2 = np.ones((n + 1,))
        for step in range(n):
            s_1[step + 1] = ds_s_1[step] * s_1[step] + s_1[step]
            s_2[step + 1] = ds_s_2[step] * s_2[step] + s_2[step]
        ss_ls_1.append(s_1)
        ss_ls_2.append(s_2)
    ss_1 = np.vstack(ss_ls_1)
    ss_2 = np.vstack(ss_ls_2)
    ss_total_1 = np.sum(lambdas.reshape((-1, 1)) * ss_1, axis=0)
    ss_total_2 = np.sum(lambdas.reshape((-1, 1)) * ss_2, axis=0)
    fig, ax = plt.subplots(ncols=2, sharey=True)
    for s in ss_ls_1:
        ax[0].plot(s, ls='--')
    for s in ss_ls_2:
        ax[1].plot(s, ls='--')
    ax[0].plot(ss_total_1, color='g')
    ax[1].plot(ss_total_2, color='g')
    ax[0].axhline(1, ls='-.', color='grey')
    ax[1].axhline(1, ls='-.', color='grey')
    plt.setp(ax, xticks=[])
    plt.show()


def draw_pdf(r, lambdas, xis, sigmas):
    s = np.linspace(0.0001, 3, 500)
    pdf_ls = []
    pdf_r_ls = []
    for i in range(len(lambdas)):
        pdf = cal_pdf(r, 1, sigmas[0], s, 1)
        pdf_ls.append(pdf)
        r_i = np.log(xis[i]) + r
        pdf_r = cal_pdf(r_i, 1, sigmas[0], s, 1)
        pdf_r_ls.append(pdf_r)
    lmm = Lognormal_mixture_model_curve(1, 1, 1, r, 0, 0, N=len(lambdas))
    lmm.set_param(p=lambdas, xi=xis, sigma=sigmas)
    vol = lmm.solve_bsm_vol([1])[0]
    pdf_bs = cal_pdf(r, 1, vol, s, 1)
    pdf_total = np.vstack(pdf_ls)
    pdf_total = np.sum(lambdas.reshape((-1, 1)) * pdf_total, axis=0)
    pdf_r_total = np.vstack(pdf_r_ls)
    pdf_r_total = np.sum(lambdas.reshape((-1, 1)) * pdf_r_total, axis=0)
    fig, ax = plt.subplots()
    ax.plot(s, pdf_total, label='Lognormal Mixture')
    ax.plot(s, pdf_r_total, label='Lognormal Mixture with Multiplicative Means')
    ax.plot(s, pdf_bs, label='BS Lognormal')
    plt.plot()
    plt.legend()
    plt.show()


def draw_vol(r, lambdas, xis, sigmas):
    s = np.linspace(0.5, 2.5, 500)
    lmm = Lognormal_mixture_model_curve(1, 1, 1, r, 0, 0, N=len(lambdas))
    lmm.set_param(p=lambdas, xi=np.array([1] * len(xis)), sigma=sigmas)
    vol1 = lmm.solve_bsm_vol([1])[0]
    lmm_mm = Lognormal_mixture_model_curve(1, 1, 1, r, 0, 0, N=len(lambdas))
    lmm_mm.set_param(p=lambdas, xi=xis, sigma=sigmas)
    vol2 = lmm_mm.solve_bsm_vol([1])[0]
    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(s, lmm.solve_bsm_vol(s), label='Lognormal Mixture')
    ax[0].axhline(vol1, ls='--', color='grey')
    ax[1].plot(s, lmm_mm.solve_bsm_vol(s), label='Lognormal Mixture with Multiplicative Means')
    ax[1].axhline(vol2, ls='--', color='grey')
    plt.plot()
    ax[0].legend()
    ax[1].legend()
    plt.show()


if __name__ == '__main__':
    lambdas = np.array([0.25, 0.25, 0.25, 0.25])
    xis = np.array([0.9, 0.9, 1.1, 1.1])
    sigmas = np.array([0.4, 0.3, 0.2, 0.3])
    r = 0.1
    draw_mixed_process(r, lambdas, sigmas)
    draw_multimeans_process(r, lambdas, xis, sigmas)
    draw_pdf(r, lambdas, xis, sigmas)
    draw_vol(r, lambdas, xis, sigmas)
