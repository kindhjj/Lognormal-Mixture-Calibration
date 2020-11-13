from pandas.tseries.offsets import BDay
from src.iostream import *
from src.utils import *
from src.USD_yield_curve import USD_yield_curve
from src.Forward import Forward
from src.Lognormal_mixture_model import Lognormal_mixture_model_surface
import logging
import sys


def log():
    logger = logging.getLogger("LognormalMixture")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        'log/solving_surface_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
        'w+')
    fmt = logging.Formatter('%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s')
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    sys.stderr.write = logger.error
    sys.stdout.write = logger.info
    pass


def output_matrix(ticker, lmms, forward, valuation_date, to_tenor=False, graph=False):
    moneyness = np.array([0.8, 0.9, 0.95, 0.975, 1., 1.025, 1.05, 1.1, 1.2])
    if to_tenor:
        tenors = [1, 2, 3, 6, 9, 12, 18, 24]
        TT = [valuation_date + pd.DateOffset(months=t) for t in tenors]
        file = '{}_output_compare_tenor'.format(ticker)
    else:
        TT = lmms.listed_dates
        file = '{}_output_compare'.format(ticker)
    given_T_forward = [forward.get_forward(t) for t in TT]
    Ks = forward.get_forward_series()[0] * moneyness
    moneyness_ls = Ks / np.array(given_T_forward).reshape((-1, 1))
    volsurfs = np.zeros_like(moneyness_ls)
    for r in range(moneyness_ls.shape[0]):
        for c in range(moneyness_ls.shape[1]):
            volsurfs[r, c] = lmms.solve_vol_from_forward_moneyness(moneyness_ls[r, c], TT[r])
    col = pd.MultiIndex.from_tuples(list(zip(['Moneyness'] * len(moneyness), moneyness)))
    df = pd.DataFrame(volsurfs, columns=col, index=TT)
    write_excel(ticker, valuation_date, df, file)

    if graph:
        moneyness = np.linspace(0.8, 1.2, 20)
        Ks = forward.get_forward_series()[0] * moneyness
        volsurfs = list(map(lambda k: lmms.solve_vol_from_strike(k, TT), Ks))
        volsurfs = np.array(volsurfs)
        plot_surface(ticker, valuation_date, TT, moneyness, volsurfs, 'volatility_surface_plot')


def print_params(ticker, params_dict):
    index = ['lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'xi_1', 'xi_2', 'xi_3', 'xi_4', 'sigma_1', 'sigma_2',
             'sigma_3', 'sigma_4']
    res_dict = dict()
    for date in params_dict:
        res_dict[date] = np.array(params_dict[date]).mean(axis=0)[:-1]
    df = pd.DataFrame(res_dict, index=index)
    df.to_csv('result/steadiness_test/{}.csv'.format(ticker))


def main(ticker, valuation_date):
    df, div_df, yc_df, p0 = get_batch_data(ticker, valuation_date, from_bbg=False, timeout=500)
    yc = USD_yield_curve(yc_df, valuation_date)
    df = data_filtering(df)
    forward = Forward(div_df, df, p0, yc, valuation_date, american=True, from_hist_div=False)

    df = solve_aprox_atm_vol(df, yc, p0, forward, valuation_date)
    df = data_filtering_for_quotes(df)
    df = df.reset_index(drop=True)
    lmms = Lognormal_mixture_model_surface(df, forward, yc, valuation_date, p0, ticker)
    output_matrix(ticker, lmms, forward, valuation_date)
    output_matrix(ticker, lmms, forward, valuation_date, to_tenor=True, graph=True)
    return lmms.print_params()


if __name__ == '__main__':
    # log()
    check_dir()
    inputs = pd.read_csv('data/input.csv')
    valuation_date = pd.to_datetime('2020-10-03') - BDay(1)
    for ticker in inputs.Ticker:
        print(ticker)
        main(ticker, valuation_date)
    for ticker in ['AAPL', 'KO']:
        params = dict()
        for date in pd.date_range('2020-10-02', '2020-10-07', freq='B'):
            params[date] = main(ticker, date)
        print_params(ticker, params)
