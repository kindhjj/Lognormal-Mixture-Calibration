import xlrd
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import stack_df
import os


def read_mkt(file):
    CP_COL = {'call': 0, 'put': 16}
    read_dict = {'strike': list(), 'putcall': list(), 'bidask': list(), 'price': list(), 'expiry': list()}
    wb = xlrd.open_workbook(file)
    for sh in wb.sheets():
        expiry = datetime.datetime(*xlrd.xldate.xldate_as_tuple(sh.cell_value(1, CP_COL['call']), datemode=0))
        for row in range(6, sh.nrows):
            for pc in CP_COL.keys():
                if sh.cell_value(row, CP_COL[pc]) == '':
                    break
                for _ in range(2):
                    read_dict['strike'].append(sh.cell_value(row, CP_COL[pc]))
                    read_dict['putcall'].append(pc)
                read_dict['bidask'].append('bid')
                read_dict['bidask'].append('ask')
                read_dict['price'].append(sh.cell_value(row, CP_COL[pc] + 2))
                read_dict['price'].append(sh.cell_value(row, CP_COL[pc] + 3))
        read_dict['expiry'] += [expiry] * (sh.nrows - 6) * 4
    df = pd.DataFrame(read_dict)
    df.loc[:, 'price'] = pd.to_numeric(df.price, errors='coerce')
    return df


def read_mkt_from_excel(file, ticker):
    CP_COL = {'call': 0, 'put': 7}
    read_dict = {'ticker': list(), 'strike': list(), 'putcall': list(), 'bidask': list(), 'price': list(),
                 'expiry': list()}
    wb = xlrd.open_workbook(file)
    sh = wb.sheet_by_index(0)
    for row in range(sh.nrows):
        underlying = sh.cell_value(row, CP_COL['call'] + 1)
        if underlying[:len(ticker)] != ticker:
            continue
        date = sh.cell_value(row, CP_COL['call'] + 1).split(' ')[1]
        date = pd.to_datetime(date, format='%m/%d/%y')
        for cp, cpcol in CP_COL.items():
            for _ in range(2):
                read_dict['ticker'].append(sh.cell_value(row, CP_COL[cp] + 1))
                read_dict['strike'].append(sh.cell_value(row, CP_COL[cp]))
                read_dict['expiry'].append(date)
                read_dict['putcall'].append(cp)
                # read_dict['implied_vol'].append(sh.cell_value(row, cpcol + 5))
            read_dict['bidask'] += ['bid', 'ask']
            read_dict['price'].append(sh.cell_value(row, cpcol + 2))
            read_dict['price'].append(sh.cell_value(row, cpcol + 3))
    df = pd.DataFrame(read_dict)
    df.loc[:, ['price']] = pd.to_numeric(df.price, errors='coerce')
    df = df.dropna()
    # df.loc[:, ['implied_vol']] = pd.to_numeric(df.implied_vol, errors='coerce')
    return df


def read_tick_quote_excel(file, ticker):
    cache_file = 'data/cache_opt_dat_{}'.format(file.split('/')[1][:-5])
    if Path(cache_file).exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=['expiry'])
        return df
    read_dict = {'ticker': list(), 'strike': list(), 'putcall': list(), 'bidask': list(), 'price': list(),
                 'expiry': list(), 'volume': list()}
    putcall_dict = {'C': 'call', 'P': 'put'}
    wb = xlrd.open_workbook(file)
    for sh in wb.sheets():
        if sh.cell_type(2, 4) == xlrd.XL_CELL_EMPTY or sh.cell_type(2, 10) == xlrd.XL_CELL_EMPTY:
            continue
        underlying = sh.cell_value(2, 0)
        putcall = putcall_dict[underlying.split(" ")[-2][0]]
        expiry = datetime.datetime(*xlrd.xldate.xldate_as_tuple(sh.cell_value(2, 2), datemode=0))
        strike = sh.cell_value(2, 1)

        read_dict['ticker'].append(underlying)
        read_dict['strike'].append(strike)
        read_dict['expiry'].append(expiry)
        read_dict['putcall'].append(putcall)
        read_dict['bidask'].append('bid')
        read_dict['price'].append(sh.cell_value(2, 4))
        read_dict['volume'].append(sh.cell_value(2, 5))

        read_dict['ticker'].append(underlying)
        read_dict['strike'].append(strike)
        read_dict['expiry'].append(expiry)
        read_dict['putcall'].append(putcall)
        read_dict['bidask'].append('ask')
        read_dict['price'].append(sh.cell_value(2, 10))
        read_dict['volume'].append(sh.cell_value(2, 11))
    df = pd.DataFrame(read_dict)
    df.loc[:, 'price'] = pd.to_numeric(df.price, errors='coerce')
    df = df.reindex(['ticker', 'putcall', 'expiry', 'strike', 'volume', 'bidask', 'price'], axis=1)
    df.to_csv(cache_file)
    return df


def read_tick_quote_excel_v2(file, ticker):
    cache_file = 'data/cache_opt_dat_{}'.format(file.split('/')[1][:-5])
    if Path(cache_file).exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=['expiry'])
        return df
    read_dict = {'ticker': list(), 'strike': list(), 'putcall': list(), 'bid': list(), 'ask': list(), 'expiry': list()}
    putcall_dict = {'C': 'call', 'P': 'put'}
    wb = xlrd.open_workbook(file)
    sh = wb.sheet_by_name('Template')
    for row in range(sh.nrows):
        if sh.cell_value(row, 0)[:len(ticker)] != ticker:
            continue
        underlying = sh.cell_value(row, 0)
        putcall = putcall_dict[underlying.split(" ")[-2][0]]
        expiry = datetime.datetime(*xlrd.xldate.xldate_as_tuple(sh.cell_value(row, 2), datemode=0))
        strike = sh.cell_value(row, 1)

        read_dict['ticker'].append(underlying)
        read_dict['strike'].append(strike)
        read_dict['expiry'].append(expiry)
        read_dict['putcall'].append(putcall)
        read_dict['bid'].append(sh.cell_value(row, 4))
        read_dict['ask'].append(sh.cell_value(row, 8))
    df = pd.DataFrame(read_dict)
    df = stack_df(df)
    df.loc[:, 'price'] = pd.to_numeric(df.price, errors='coerce')
    df = df.dropna().reindex(['ticker', 'putcall', 'expiry', 'strike', 'bidask', 'price'], axis=1)
    df.to_csv(cache_file)
    return df


def read_dvd(file):
    wb = xlrd.open_workbook(file)
    sh = wb.sheet_by_index(0)
    read_dict = {'date': list(), 'dividend': list()}
    for row in range(3, sh.nrows):
        read_dict['date'].append(datetime.datetime(*xlrd.xldate.xldate_as_tuple(sh.cell_value(row, 0),
                                                                                datemode=0)))
        read_dict['dividend'].append(sh.cell_value(row, 2))
    df = pd.DataFrame(read_dict)
    return df


def read_gvt_bd_hk(file):
    df = pd.read_excel(file)
    return df


def read_mkt_yield_curve(file):
    df = pd.read_excel(file, parse_dates=[0, 1])
    df = df[['Maturity Date', 'Zero Rate', 'Discount']]
    return df


def _transform_date(date_str):
    mdy = date_str.split('/')
    return pd.to_datetime('20{}{}{}'.format(mdy[2], mdy[0], mdy[1]), format='%Y%m%d')


def _check_cach_path():
    if not Path('cached_data/').exists():
        os.mkdir('cached_data/')


# read option quotes from bbg terminal
def get_option_data(ticker, query_date_raw, timeout=1000):
    _check_cach_path()
    query_date = query_date_raw
    start_time = '{}T{}'.format(query_date.strftime("%Y-%m-%d"), "19:44:00")
    end_time = '{}T{}'.format(query_date.strftime("%Y-%m-%d"), "19:45:00")
    ticker += ' Equity'
    # check cached file
    file = 'cached_data/opt_price_{}_{}.csv'.format(ticker.replace('/', ''), query_date_raw.strftime("%Y%m%d"))
    if Path(file).exists():
        df_res = pd.read_csv(file)
        df_res['expiry'] = pd.to_datetime(df_res.expiry)
        return df_res

    print('Fetching ticker: ' + ticker)
    con = pdblp.BCon(debug=False, timeout=3000)
    con.start()
    # get option chain names
    df_ticker = con.bulkref(ticker, 'OPT_CHAIN', ovrds=[('OPTION_CHAIN_OVERRIDE', 'A')])
    con.stop()
    opt_info = df_ticker['value'].map(lambda x: x.split(' '))
    expiry = opt_info.map(lambda x: _transform_date(x[-3]))
    putcall_map = {'P': 'put', 'C': 'call'}
    putcall = opt_info.map(lambda x: putcall_map[x[-2][0]])
    strike = opt_info.map(lambda x: float(x[-2][1:]))
    con = pdblp.BCon(debug=False, timeout=timeout)
    con.start()
    df_price = pd.DataFrame({'ticker': df_ticker.value, 'expiry': expiry, 'strike': strike, 'putcall':
        putcall}).set_index('ticker')
    df_price['bid'] = [np.nan] * len(df_price)
    df_price['ask'] = [np.nan] * len(df_price)
    print('{} options to fetch.'.format(len(df_price)))

    # get prices
    for tikr in df_price.index:
        for bidask in ['BID', 'ASK']:
            try:
                print(ticker)
                df_p = con.bdib(tikr, start_time, end_time, bidask, interval=1, elms=[('gapFillInitialBar', True)])
                if len(df_p):
                    df_price.loc[tikr, bidask.lower()] = df_p.close[0]
            except Exception as e:
                print(e)

    df_price = df_price.reset_index()
    df_price = stack_df(df_price)
    df_price.to_csv(file, index=False)
    return df_price


def get_proj_dividend(ticker):
    _check_cach_path()
    ticker += ' Equity'
    datpath = 'cached_data/proj_div_{}.csv'.format(ticker.replace('/', ''))
    if Path(datpath).exists():
        df_res = pd.read_csv(datpath)
        df_res['date'] = pd.to_datetime(df_res.date)
        return df_res
    con = pdblp.BCon(debug=False, timeout=3000)
    con.start()
    df = con.bulkref(ticker, 'BDVD_ALL_PROJECTIONS')
    if not len(df.dropna()):
        df = pd.DataFrame(dict(date=[pd.to_datetime('2030-12-31')], dividend=[0]))
    else:
        df = df[['name', 'value']]
        cols = np.unique(df.name)
        parse_dict = {c: df[df.name == c].value.to_list() for c in cols}
        df = pd.DataFrame(parse_dict)
        df = df.rename(columns={'Amount Per Share': 'dividend', 'Projected Ex-Date': 'date'})
        df = df[['Ex-Date', 'dividend']]
        df = df.rename(columns={'Ex-Date': 'date'})
        df['date'] = pd.to_datetime(df.date, format='%Y-%m-%d')
    df.to_csv(datpath, index=False)
    return df


def get_hist_price(ticker, query_date_raw, timeout=1000):
    _check_cach_path()
    ticker += ' Equity'
    query_date = query_date_raw
    start_time = '{}T{}'.format(query_date.strftime("%Y-%m-%d"), "19:44:00")
    end_time = '{}T{}'.format(query_date.strftime("%Y-%m-%d"), "19:45:00")
    file = 'cached_data/spot_{}_{}'.format(ticker.replace('/', ''), query_date_raw.strftime("%Y%m%d"))
    if Path(file).exists():
        with open(file, 'r') as f:
            price = f.readline()
        price = float(price)
        return price
    con = pdblp.BCon(debug=False, timeout=timeout)
    con.start()
    df_bid = con.bdib(ticker, start_time, end_time, 'BID', interval=1, elms=[('gapFillInitialBar', True)])
    df_ask = con.bdib(ticker, start_time, end_time, 'ASK', interval=1, elms=[('gapFillInitialBar', True)])
    price = (df_bid.close[0] + df_ask.close[0]) / 2
    with open(file, 'w') as f:
        f.write('{:.3f}'.format(price))
    return price


# get batch data
def get_batch_data(ticker, query_date, from_bbg=False, timeout=1000):
    data_dir = 'data/'
    qdata_str = query_date.strftime('%Y%m%d')
    if not from_bbg:
        ticker_dvd = pd.read_csv('data/tickers.csv')
        ticker_dvd = ticker_dvd[ticker_dvd.Ticker == ticker].squeeze()
        ticker_for_dvd = '{} {}'.format(ticker_dvd[0], ticker_dvd[1])
        p_file = '{}/{}_{}.xlsx'.format(data_dir, ticker, qdata_str)
        price_df = read_mkt_from_excel(p_file, ticker)
        dividend_df = _read_dividend_batch(data_dir, ticker_for_dvd)
        yc_df = _read_yield_curve_batch(data_dir, qdata_str)
        spot = _read_spot_batch(data_dir, ticker, qdata_str)
    else:
        with os.add_dll_directory(os.path.abspath('venv/Library/bin')):
            import pdblp
        price_df = get_option_data(ticker, query_date, timeout=timeout)
        dividend_df = get_proj_dividend(ticker)
        data_dir = 'data/'
        yc_df = _read_yield_curve_batch(data_dir, qdata_str)
        spot = get_hist_price(ticker, query_date, timeout=timeout)
    return price_df, dividend_df, yc_df, spot


def get_batch_data_without_price(ticker, query_date, from_bbg=False, timeout=1000):
    data_dir = 'data/{}/'.format(ticker)
    qdata_str = query_date.strftime('%Y%m%d')
    if not from_bbg:
        dividend_df = _read_dividend_batch(data_dir, ticker)
        yc_df = _read_yield_curve_batch(data_dir, qdata_str)
        spot = _read_spot_batch(data_dir, ticker, qdata_str)
    else:
        dividend_df = get_proj_dividend(ticker)
        data_dir = 'data/'
        yc_df = _read_yield_curve_batch(data_dir, qdata_str)
        spot = get_hist_price(ticker, query_date, timeout=timeout)
    return dividend_df, yc_df, spot


def _read_price_data_batch(data_dir, ticker, query_date):
    read_dict = {'ticker': list(), 'strike': list(), 'putcall': list(), 'bid': list(), 'ask': list(), 'expiry': list()}
    wb = xlrd.open_workbook('{}{}_{}.xlsx'.format(data_dir, ticker, query_date))
    sh = wb.sheet_by_name('Template')
    for row in range(1, sh.nrows):
        underlying = sh.cell_value(row, 0)
        putcall = sh.cell_value(row, 1)
        expiry = datetime.datetime(*xlrd.xldate.xldate_as_tuple(sh.cell_value(row, 3), datemode=0))
        strike = sh.cell_value(row, 2)

        read_dict['ticker'].append(underlying)
        read_dict['strike'].append(strike)
        read_dict['expiry'].append(expiry)
        read_dict['putcall'].append(putcall.lower())
        read_dict['bid'].append(sh.cell_value(row, 4))
        read_dict['ask'].append(sh.cell_value(row, 5))
    df = pd.DataFrame(read_dict)
    df = stack_df(df)
    df.loc[:, 'price'] = pd.to_numeric(df.price, errors='coerce')
    df = df.dropna().reindex(['ticker', 'putcall', 'expiry', 'strike', 'bidask', 'price'], axis=1)
    return df


def _read_dividend_batch(data_dir, ticker):
    file = '{}BDVD_dividend.csv'.format(data_dir)
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    col = 'proj_div_' + ticker + ' Equity'
    df_t = df[[col]].rename(columns={col: 'dividend'}).dropna().reset_index()
    return df_t


def _read_yield_curve_batch(data_dir, query_date):
    file = '{}mkt_yield_curve_{}.xlsx'.format(data_dir, query_date)
    df = pd.read_excel(file)
    df['Maturity Date'] = df['Maturity Date'].map(lambda x: pd.to_datetime(x, format='%m/%d/%Y'))
    return df


def _read_spot_batch(data_dir, ticker, query_date):
    file = '{}spot_{}_{}.txt'.format(data_dir, ticker, query_date)
    with open(file, 'r') as f:
        price = f.readline()
    price = float(price)
    return price


def _check_dir(path):
    if not Path(path).exists():
        os.makedirs(path)


def write_excel(ticker, date, df, file_name):
    _check_dir('result/{}/{}'.format(date.strftime('%Y%m%d'), ticker))
    with pd.ExcelWriter('result/{}/{}/{}.xlsx'.format(date.strftime('%Y%m%d'), ticker, file_name),
                        datetime_format='dd/mm/yyy') as excel_writer:
        df.to_excel(excel_writer)


def plot_surface(ticker, date, x, y, z, file_name):
    _check_dir('result/{}/{}'.format(date.strftime('%Y%m%d'), ticker))
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Sunsetdark')])
    fig.update_layout(scene_aspectmode='manual',
                      scene_aspectratio=dict(x=1, y=1, z=0.5))
    fig.write_html('result/{}/{}/{}.html'.format(date.strftime('%Y%m%d'), ticker, file_name))

def check_dir():
    _check_dir('cached_data')
    _check_dir('result/steadiness_test')