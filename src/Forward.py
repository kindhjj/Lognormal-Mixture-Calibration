import bisect
import datetime
import numpy as np
import pandas as pd
from functools import reduce
from dateutil.relativedelta import relativedelta
from src.American_solver import American_solver, European_option


def _impliedforward(c1, c2, p1, p2, k1, k2, yc):
    c1 = np.mean(c1, axis=1)
    p1 = np.mean(p1, axis=1)
    c1 = c1[p1.index]
    p1 = p1[c1.index]
    f1 = np.mean(c1 - p1 + k1 * yc.df(c1.index))
    c2 = np.mean(c2, axis=1)
    p2 = np.mean(p2, axis=1)
    f2 = np.mean(c2 - p2 + k2 + yc.df(c2.index))
    return (f1 + f2) / 2


def _cal_dividend_from_yield(dividend_yield, expiry_date, valuation_date, p0):
    return dividend_yield * (expiry_date - valuation_date).days / 360 * p0


def _cal_dividend(f, yc, expiry_date, valuation_date, p0):
    dividend_yield = _cal_dividend_yield(f, yc, expiry_date, valuation_date, p0)
    return _cal_dividend_from_yield(dividend_yield, expiry_date, valuation_date, p0)


def _cal_dividend_yield_from_div(d, expiry_date, valuation_date, p0):
    if d is np.nan:
        d = 0
    return d / (expiry_date - valuation_date).days * 360 / p0


def _cal_dividend_yield(f, yc, expiry_date, valuation_date, p0):
    return (p0 / yc.df(expiry_date) / f - 1) * 360 / (expiry_date - valuation_date).days


def _cal_impl_fwd(dividend_yield, yc, expiry_date, valuation_date, p0):
    return p0 / yc.df(expiry_date) / (1 + dividend_yield * (expiry_date - valuation_date).days / 360)


def _gen_price_matrix(df, k1, k2):
    c1 = df[(df.strike == k1) & (df.putcall == 'call')].pivot(index='expiry', columns='bidask', values='price')
    c2 = df[(df.strike == k2) & (df.putcall == 'call')].pivot(index='expiry', columns='bidask', values='price')
    p1 = df[(df.strike == k1) & (df.putcall == 'put')].pivot(index='expiry', columns='bidask', values='price')
    p2 = df[(df.strike == k2) & (df.putcall == 'put')].pivot(index='expiry', columns='bidask', values='price')
    return c1, c2, p1, p2


def _impliedforward(c1, c2, p1, p2, k1, k2, yc):
    c1.insert(2, 'mean', np.mean(c1, axis=1))
    p1.insert(2, 'mean', np.mean(p1, axis=1))
    # c1 = c1[p1.index]
    # p1 = p1[c1.index]
    f1 = c1 - p1 + k1 * yc.df(c1.index[0])
    c2.insert(2, 'mean', np.mean(c2, axis=1))
    p2.insert(2, 'mean', np.mean(p2, axis=1))
    f2 = c2 - p2 + k2 * yc.df(c2.index[0])
    return np.mean((f1 + f2) / 2, axis=0)


def _de_americanize_vol(price, yc, p0, dividend_yield, k, callput, valuation_date, expiry_date):
    american_option = American_solver(yc, p0, dividend_yield, k, callput, valuation_date, expiry_date, valuation_date)
    ame_vol = american_option(price)
    eur_option = European_option(yc, p0, dividend_yield, k, callput, valuation_date, expiry_date, valuation_date)
    return eur_option(ame_vol)


def _de_americanize(price, yc, p0, dividend_yield, k, callput, valuation_date):
    for r in range(price.shape[0]):
        for c in range(price.shape[1]):
            price.iloc[r, c] = _de_americanize_vol(price.iloc[r, c], yc, p0, dividend_yield, k, callput, valuation_date,
                                                   price.index[r])
    return price


def _interval_intersection(pairs_list):
    pairs_list = [[min(a), max(a)] for a in pairs_list]
    res = list(zip(*pairs_list))
    res[0] = max(res[0])
    res[1] = min(res[1])
    if res[0] > res[1]:
        return np.mean(pairs_list[0])
    else:
        return np.mean(res)


def _adjust_imp_dividend(div_df, forward_div, exp_dates, valuation_date, p0):
    forward_div_df = pd.DataFrame(forward_div,
                                  columns=['forward', 'dividend_yield', 'bid_div', 'ask_div'],
                                  index=exp_dates)
    forward_div_df = forward_div_df.dropna()
    # div_df_cumsum = div_df.copy()
    # div_df_cumsum.dividend = div_df_cumsum.dividend.cumsum()
    for i in range(div_df.shape[0] - 1):
        if div_df.date[i] > forward_div_df.index[-1]:
            break
        else:
            df_temp = forward_div_df.loc[div_df.date[i]:div_df.date[i + 1], ['bid_div', 'ask_div']]
            if df_temp.shape[0] > 1:
                div_new = _interval_intersection([r for _, r in df_temp.iterrows()])
                new_series = pd.Series(_cal_dividend_yield_from_div(div_new, df_temp.index, valuation_date, p0),
                                       index=df_temp.index)
                for index, value in new_series.items():
                    forward_div_df.loc[index, 'dividend_yield'] = value

    forward_div_df['implied_dividend'] = pd.Series(_cal_dividend_from_yield(forward_div_df.dividend_yield,
                                                                            forward_div_df.index,
                                                                            valuation_date,
                                                                            p0),
                                                   index=forward_div_df.index)
    return forward_div_df


def _processforward(yc, p0, quote_df, dividend_yield, expiry_date, valuation_date=datetime.datetime.today(),
                    american=False):
    f = _cal_impl_fwd(dividend_yield, yc, expiry_date, valuation_date, p0)
    grouped = quote_df.groupby('putcall')
    Kset1 = set(grouped.get_group('call')['strike'])
    Kset2 = set(grouped.get_group('put')['strike'])
    Ks = np.sort(list(Kset1.intersection(Kset2)))
    count = 0
    while True:
        count += 1
        if (not len(Ks)) or (count == 10):
            return None, None, None, None
        i1 = bisect.bisect_left(Ks, f) - 1
        if i1 + 1 == len(Ks):
            return None, None, None, None
        k1 = Ks[i1]
        k2 = Ks[i1 + 1]
        c1, c2, p1, p2 = _gen_price_matrix(quote_df, k1, k2)
        if american:
            c1 = _de_americanize(c1, yc, p0, dividend_yield, k1, 1, valuation_date)
            c2 = _de_americanize(c2, yc, p0, dividend_yield, k2, 1, valuation_date)
            p1 = _de_americanize(p1, yc, p0, dividend_yield, k1, -1, valuation_date)
            p2 = _de_americanize(p2, yc, p0, dividend_yield, k2, -1, valuation_date)
        f_new = _impliedforward(c1, c2, p1, p2, k1, k2, yc)
        if k1 < f_new['mean'] < k2 and abs(f_new['mean'] / f - 1) < 0.001:
            f = f_new['mean']
            dividend_yield = 0 if dividend_yield == 0 else _cal_dividend_yield(f, yc, expiry_date, valuation_date, p0)
            dividend = [0, 0] if dividend_yield == 0 else _cal_dividend(f_new[['bid', 'ask']],
                                                                        yc,
                                                                        expiry_date,
                                                                        valuation_date,
                                                                        p0).tolist()
            break
        f = f_new['mean']
        dividend_yield = 0 if dividend_yield == 0 else _cal_dividend_yield(f, yc, expiry_date, valuation_date, p0)
    return f, dividend_yield, dividend[0], dividend[1]

# forecasting future dividend with historical dividend
def _hist_div(div_df, valuation_date):
    dfn = []
    for y in range(20):
        dfi = div_df.copy()
        dfi.date = div_df.date.map(lambda x: x + relativedelta(years=y))
        dfn.append(dfi)
    new_df = reduce(lambda x, y: pd.concat([x, y], axis=0, ignore_index=True), dfn)
    new_df = new_df[new_df.date > valuation_date].reset_index()
    return new_df


class Forward:
    def __init__(self, div_df, price_df, p0, yc, valuation_date, american=True, from_hist_div=False):
        self._date = []
        forward_div = []
        if from_hist_div:
            self._div_df = _hist_div(div_df, valuation_date)
        else:
            self._div_df = div_df
        self._div_df['yearmonth'] = self._div_df.date.map(lambda x: x.strftime('%Y%m'))
        for expiry in pd.to_datetime(np.unique(price_df.expiry)):
            df_temp = price_df[price_df.expiry == expiry]
            if len(np.unique(df_temp.putcall)) < 2:
                continue
            self._date.append(expiry)
            div = self._div_df[self._div_df.date <= expiry]
            dividend_yield = _cal_dividend_yield_from_div(div.dividend.cumsum().max(), expiry, valuation_date, p0)
            forward_div.append(_processforward(yc, p0, df_temp, dividend_yield, expiry, valuation_date, american))
        forward_div_df = _adjust_imp_dividend(self._div_df, forward_div, self._date, valuation_date, p0)
        self._yc = yc
        self._spot = p0
        self._valuation_date = valuation_date
        self.forward = forward_div_df.forward
        self.dividend_yield = forward_div_df.dividend_yield
        self.implied_dividend = forward_div_df.implied_dividend
        self._date = forward_div_df.index.tolist()

    # get interpolated forward price at specific date
    def get_forward(self, date):
        if date in self._date:
            return self.forward[date]
        else:
            f_i = bisect.bisect_left(self._date, date) - 1
            if f_i < 0:
                forw = _cal_impl_fwd(0, self._yc, date, self._valuation_date, self._spot)
                return forw
            else:
                div_yield = self.get_dividend_yield(date)
                forw = _cal_impl_fwd(div_yield, self._yc, date, self._valuation_date, self._spot)
                return forw

    # get forward series generated by quote prices (no interpolation)
    def get_forward_series(self):
        return self.forward

    def get_date(self):
        return self._date

    # get interpolated dividend yield
    def get_dividend_yield(self, date):
        if date in self._date:
            return self.dividend_yield[date]
        else:
            f_i = bisect.bisect_left(self._date, date) - 1
            if f_i < 0:
                forw = _cal_impl_fwd(0, self._yc, date, self._valuation_date, self._spot)
                div_yield = _cal_dividend_yield(forw, self._yc, date, self._valuation_date, self._spot)
                return div_yield
            elif f_i + 1 >= len(self.forward):
                total_div = self.implied_dividend[-1]
                div_date = self._date[-1]
                while True:
                    div_date += relativedelta(months=1)
                    if div_date > date:
                        break
                    elif div_date.strftime('%Y%m') in self._div_df.yearmonth.array:
                        total_div += self._div_df.loc[
                            self._div_df.yearmonth == div_date.strftime('%Y%m'), 'dividend'].squeeze()
            else:
                div_to_allocate = self.implied_dividend[f_i + 1] - self.implied_dividend[f_i]
                div_base = dict(date=list(), dividend=list())
                div_date = self._date[f_i]
                while True:
                    div_date += relativedelta(months=1)
                    if div_date > self._date[f_i + 1]:
                        break
                    elif div_date.strftime('%Y%m') in self._div_df.yearmonth.array:
                        div_base['date'].append(self._div_df.loc[self._div_df.yearmonth == div_date.strftime('%Y%m'),
                                                                 'date'].squeeze())
                        div_base['dividend'].append(
                            self._div_df.loc[self._div_df.yearmonth == div_date.strftime('%Y%m'), 'dividend'].squeeze())
                div_base = pd.DataFrame(div_base)
                div_base.dividend = div_base.dividend / div_base.dividend.sum()
                if len(div_base):
                    allocated_div = div_to_allocate * div_base[div_base.date <= date]['dividend'].sum()
                else:
                    allocated_div = 0
                total_div = self.implied_dividend[f_i] + allocated_div
            div_yield = _cal_dividend_yield_from_div(total_div, date, self._valuation_date, self._spot)
            return div_yield


if __name__ == '__main__':
    from src.iostream import *
    from src.utils import data_filtering
    from src.USD_yield_curve import USD_yield_curve

    ticker = 'XOM US'
    yc_file = '../data/mkt_yield_curve_20200817_XOM.xlsx'
    quote_file = '../data/xom_20200817.xlsx'
    now = pd.to_datetime(datetime.date(2020, 8, 17))
    last = now - pd.offsets.BDay(1)
    # df = get_option_data(ticker, last, now)
    df = read_tick_quote_excel(quote_file, 'XOM')
    div_df = get_proj_dividend(ticker)
    valuation_date = now
    yc = USD_yield_curve(read_mkt_yield_curve(yc_file), valuation_date)
    df = data_filtering(df)
    # p0 = get_current_price(ticker)
    p0 = 42.63
    f = Forward(div_df, df, p0, yc, valuation_date, True)
    print(f.forward)
    # print(f._div)
    print(f.dividend_yield)
    print(f.implied_dividend)
