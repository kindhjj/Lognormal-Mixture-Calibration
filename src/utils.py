import numpy as np
import pandas as pd
from functools import reduce
from scipy.stats import norm
import sklearn
from src.American_solver import American_solver

CALL_PUT = {'call': 1, 'put': -1}


def data_filtering(df):
    index_col = [c for c in df.columns if c not in ['bidask', 'price']]
    df_temp = df.set_index(index_col).pivot(columns='bidask').droplevel(0, axis=1)
    df_temp = df_temp[~(df_temp.bid.isna() | df_temp.ask.isna())]  # scope out NA quotes

    # TODO: filter out quotes older than 3 days
    df_temp['spread'] = df_temp.ask - df_temp.bid
    df_temp['abs_spread'] = np.abs(df_temp.spread)
    df_temp['mid'] = (df_temp.bid + df_temp.ask) / 2
    df_temp = df_temp[df_temp.abs_spread >= 0.001]  # filter out too small spreads
    df_temp = df_temp[df_temp.ask <= 3 * df_temp.bid]  # filter out too big spreads
    df_temp_for_filter = df_temp.groupby(level=['expiry'])['abs_spread'].agg(['std', 'mean']).rename(
        columns={'std': 'spread_std', 'mean': 'spread_mean'}).reset_index()
    df_temp = df_temp.reset_index().merge(df_temp_for_filter, how='left', on='expiry').set_index(index_col)
    df_temp = df_temp[
        (df_temp.abs_spread <= df_temp.spread_mean + 3 * df_temp.spread_std) & (
                df_temp.abs_spread >= df_temp.spread_mean - 3 * df_temp.spread_std)]  # filter out too big spreads
    # idx = pd.IndexSlice
    # time_mask_in1year = df_temp.loc[idx[:, :valuation_date + pd.to_timedelta('1y'), :], :]
    # time_mask_over1year = df_temp.loc[idx[:, valuation_date + pd.to_timedelta('1y'):, :], :]
    # filter_mask = (time_mask_in1year.abs_spread / time_mask_in1year.mid <= 0.0009) | (
    #         time_mask_over1year.abs_spread <= 0.021)  # filter out too big spreads
    # df_temp = df_temp[filter_mask]

    df_temp = stack_df(df_temp[['bid', 'ask']].reset_index())
    return df_temp


def data_filtering_for_quotes(df):
    df = data_filtering(df)
    fifteen_median_vol = 15 * np.median(df.implied_vol)
    onefifteenth_median_vol = 1 / 15 * np.median(df.implied_vol)
    df = df[(onefifteenth_median_vol <= df.implied_vol) & (df.implied_vol <= fifteen_median_vol)]
    df = df.merge(
        df.groupby('expiry')['price'].count().reset_index().rename(columns={'price': 'cts'}), how='left',
        on='expiry')
    df = df[df.cts >= (0.2 * df.cts.median())]
    return df


def calculate_mid_price(df):
    temp_df = pivot_df(df)
    index_col = [c for c in temp_df.columns if c not in ['bid', 'ask']]
    temp_df = temp_df.set_index(index_col)
    temp_df['price'] = temp_df.mean(axis=1)
    temp_df = temp_df.reset_index()
    return temp_df


def pivot_df(df):
    index_col = [c for c in df.columns if c not in ['bidask', 'price']]
    temp_df = df.set_index(index_col).pivot(columns='bidask').droplevel(0, axis=1).reset_index()
    return temp_df


def stack_df(df):
    index_col = [c for c in df.columns if c not in ['bid', 'ask']]
    temp_df = df.melt(id_vars=index_col, var_name='bidask', value_name='price')
    return temp_df


def calc_bsm_price(r, d, p0, k, vol, T, callput):
    if isinstance(callput, str):
        callput = CALL_PUT[callput.lower()]
    d1 = (np.log(p0 / k) + (r - d + 0.5 * np.square(vol) * T)) / vol / np.sqrt(T)
    d2 = d1 - vol * np.sqrt(T)
    price = (p0 * np.exp(-d * T) * norm.cdf(d1 * callput) - np.exp(-r * T) * k * norm.cdf(d2 * callput)) * callput
    return price


def solve_aprox_atm_vol(df, yc, p0, forward, valuation_date):
    dfn = []
    df_temp = pivot_df(df)
    df_temp['price'] = (df_temp.bid + df_temp.ask) / 2
    df_expiries = df_temp.expiry[~df_temp.expiry.duplicated()].sort_values().reset_index(drop=True)
    for t in df_expiries:
        df_sub_time = df_temp[df_temp.expiry == t].reset_index(drop=True)
        k_nearest = df_sub_time.strike[np.argmin(np.abs(df_sub_time.strike - p0))]
        df_t_k = df_sub_time[(df_sub_time.strike == k_nearest)].reset_index(drop=True).iloc[0].squeeze()
        try:
            implvol = American_solver(yc, p0, forward.get_dividend_yield(t), df_t_k['strike'], df_t_k['putcall'],
                                      valuation_date,
                                      df_t_k['expiry'], valuation_date)(df_t_k['price'])
        except Exception as e:
            implvol = -1
        df_sub_time['implied_vol'] = implvol
        dfn.append(df_sub_time)
    df_temp = reduce(lambda x, y: pd.concat([x, y], ignore_index=True), dfn)
    df_temp = df_temp.drop('price', axis=1)
    df_temp = df_temp[df_temp.implied_vol > 0]
    df_temp = stack_df(df_temp)
    return df_temp


def random_cut_df(df, n, spot):
    if len(df) <= n:
        return df
    mask_atm = (df.strike > (spot * 0.95)) & (df.strike < (spot * 1.05))
    mask_otm = ~mask_atm
    df_atm = df[mask_atm].reset_index(drop=True)
    if len(df_atm) >= n:
        n = n // 2
        throw_loc = sklearn.utils.resample(list(range(len(df_atm))), n_samples=len(df_atm) - n, replace=False)
        df_atm = df_atm.drop(throw_loc)
    df_otm = df[mask_otm].reset_index(drop=True)
    throw_loc = sklearn.utils.resample(list(range(len(df_otm))), n_samples=len(df_otm) - n, replace=False)
    df_otm = df_otm.drop(throw_loc)
    df_new = pd.concat([df_atm, df_otm], ignore_index=True).sort_values('expiry').reset_index(drop=True)
    return df_new


def trim_x(x_array, N):
    p = x_array[:N]
    xi = x_array[N:2 * N]
    sigma = x_array[2 * N:3 * N]
    p = np.array([max(i, 1e-3) for i in p])
    p = p / sum(p)
    p[-1] = p[-1] - 1e-5
    xi = xi / p.dot(xi)
    new_x = np.concatenate([p, xi, sigma, [1e-5]], axis=0)
    return new_x


def trim_sigma(x_array, lb, N):
    for i in range(N):
        if x_array[2 * N + i] < lb[2 * N + i]:
            x_array[2 * N + i] = lb[2 * N + i]
    return x_array


def cal_pdf(r, t, vol, s, s_0):
    return 1 / (s * vol * np.sqrt(2 * np.pi * t)) * np.exp(
        -1 / 2 / np.square(vol) / t * np.square(np.log(s / s_0) - r * t + 0.5 * np.square(vol) * t))
