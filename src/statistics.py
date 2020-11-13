import pandas as pd
import numpy as np


def mse(compute, market):
    tenor = ['1M', '2M', '3M', '6M', '9M', '1Y', '18M', '2Y']
    compute['tenor'] = tenor
    compute = compute.set_index('tenor', drop=True)
    compute = compute.stack()
    market = market.stack()
    df = compute.to_frame('model')
    df['market'] = market / 100
    df['diff'] = np.square(df['market'] - df['model'])
    df = df[['diff']].reset_index().groupby('tenor')['diff'].mean()
    return df


def read_compute(ticker):
    header = ['tenor', 0.8, 0.9, 0.95, 0.975, 1, 1.025, 1.05, 1.1, 1.2]
    df = pd.read_excel('result/20201002/{0}/{0}_output_compare_tenor.xlsx'.format(ticker), 'Sheet1', names=header)
    df = df.iloc[2:, :].set_index('tenor')
    return df


def read_market(ticker):
    df = pd.read_excel('result/bbg_surface/{0}.xlsx'.format(ticker), 'Sheet1')
    df = df.iloc[2:, :].set_index('Expiry').drop([' Exp Date', 'ImpFwd'], axis=1)
    return df


def concat_df(ticker_row, mse_row):
    le = len(mse_row)
    df = mse_row.reset_index()
    df['Ticker'] = [ticker_row['Ticker']] * le
    df['Class'] = [ticker_row['Class']] * le
    return df


# calculate MSD result
def cal_MSD():
    from functools import reduce

    tenor = ['1M', '2M', '3M', '6M', '9M', '1Y', '18M', '2Y']
    tks = pd.read_csv('data/input.csv')
    mse_ls = []
    for index, row in tks.iterrows():
        mse_row = mse(read_compute(row['Ticker']), read_market(row['Ticker']))
        mse_ls.append(concat_df(row, mse_row))
    mse_df = reduce(lambda x, y: pd.concat([x, y], axis=0), mse_ls)
    mse_table = mse_df.groupby(['Class', 'tenor'])['diff'].mean().unstack().reindex(columns=tenor)

    mse_table = mse_table * 1000
    print(mse_table.to_string(float_format='{:.2f}'.format))
    print(mse_df.groupby('Class')['diff'].mean() * 1000)


def read_full_price(read_data_func):
    from functools import reduce
    tks = pd.read_csv('data/input.csv')
    df_ls = []
    for index, row in tks.iterrows():
        dfi = read_data_func(row['Ticker'])
        dfi['underlying'] = [row['Ticker']] * len(dfi)
        dfi['class'] = [row['Class']] * len(dfi)
        df_ls.append(dfi)
    return reduce(lambda x, y: pd.concat([x, y], axis=0), df_ls)


# option statistics
def option_stat(valuation_date):
    from src.iostream import read_mkt_from_excel
    from src.utils import pivot_df
    stats_df = read_full_price(lambda x: read_mkt_from_excel('data/{}_{}.xlsx'.format(x, valuation_date), x))
    result_dict = dict()
    result_dict['N'] = stats_df.groupby('class')['price'].count()
    stats_pivot_df = pivot_df(stats_df)
    result_dict['std'] = stats_pivot_df.groupby('class')[['bid', 'ask']].std().mean(axis=1)
    stats_pivot_df['mid'] = (stats_pivot_df['bid'] + stats_pivot_df['ask']) / 2
    stats_pivot_df['spread'] = np.abs(stats_pivot_df['bid'] - stats_pivot_df['ask']) / stats_pivot_df['mid']
    result_dict['spread'] = stats_pivot_df.groupby('class')['spread'].mean() * 100
    result_dict['Ks'] = \
        stats_pivot_df.groupby(['class', 'underlying', 'expiry'])['strike'].count().reset_index().groupby(
            ['class', 'underlying'])['strike'].mean().reset_index().groupby('class')['strike'].mean()
    result_dict['Ts'] = \
        stats_pivot_df.groupby(['class', 'underlying', 'expiry'])['strike'].count().reset_index().groupby(
            ['class', 'underlying'])['strike'].count().reset_index().groupby('class')['strike'].mean()

    df = pd.DataFrame(result_dict)
    print(df.to_string(float_format='{:.2f}'.format))


# statistics of option price after pre-process
def option_price_stat():
    from scipy.stats import skew, kurtosis
    stats_df = read_full_price(lambda x: pd.read_csv('cached_data/{}_processed.csv'.format(x), index_col=0))
    stats_df = stats_df.groupby(['class', 'putcall'])['price'].agg(
        [np.min, np.max, np.mean, np.median, np.std, skew, kurtosis]).reset_index()
    print(stats_df.to_string(float_format='{:.4f}'.format))


if __name__ == '__main__':
    valuation_date = '20201002'
    option_stat(valuation_date)
    option_price_stat() # must run 'main.py' first to produce all processed data necessary for this function.
    cal_MSD()
