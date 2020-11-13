from src.iostream import *
from src.utils import *
from src.Forward import _de_americanize_vol, Forward
from src.USD_yield_curve import USD_yield_curve
from src.Lognormal_mixture_model import Lognormal_mixture_model_curve, Lognormal_mixture_model_surface


# from LNM_extended import Lognormal_mixture_model_surface_extended


def test_volcurve(df_t, T, dividend_yield, lmms=None, n=0):
    print(T)
    option_mask = ((df_t.putcall == 'call') & (df_t.strike >= f / yc.df(T))) | (
            (df_t.putcall == 'put') & (df_t.strike < f / yc.df(T)))
    # option_mask = (df_t.putcall == 'call') & (df_t.strike >= f / yc.df(T))
    df_t = df_t[option_mask].reset_index(drop=True)
    for row in range(df_t.shape[0]):
        df_t.loc[row, 'price'] = _de_americanize_vol(df_t.loc[row, 'price'], yc, p0, dividend_yield,
                                                     df_t.loc[row, 'strike'],
                                                     df_t.loc[row, 'putcall'],
                                                     valuation_date, df_t.loc[row, 'expiry'])

    df_t = calculate_mid_price(df_t)
    Ks = df_t['strike']
    x0atm_sig = df_t.implied_vol[0]
    Prices = df_t.price

    r = -np.log(yc.df(T))
    T = (T - pd.to_datetime(valuation_date)).days / 365
    r /= T
    Callput = df_t.putcall
    if lmms is None:
        lmm = Lognormal_mixture_model_curve(T, p0, f, r, dividend_yield, valuation_date, N=4)
        # success = lmm.solve_global(np.array(Ks), np.array(Prices), Callput)
        x0 = np.array(
            [0.4, 0.3, 0.2, 0.1 - 1e-7, 1. + 1.001e-6, 0.8, 1.3, 1, x0atm_sig, x0atm_sig, x0atm_sig, x0atm_sig,
             1e-7])
        # x0 = np.array([0.3, 0.3, 0.2, 0.2, 1, 1, 1, 1, x0atm_sig, x0atm_sig, x0atm_sig, x0atm_sig, 0])
        success = lmm.solve_param(np.array(Ks), np.array(Prices), Callput, x0, 0, False, x0atm_sig)

        print(success)
    else:
        lmm = lmms.curves[n]
    # lmm.solve_global(np.array(Ks), np.array(Asks))
    # lmm.set_param(np.array([0.4, 0.6]), np.array([0.25, 0.25]), np.array([0.1, 0.3]))
    print(lmm.p)
    print(lmm.xi)
    print(lmm.sigma)
    print(lmm.solve_bsm_vol(p0))
    KK = 3122.5
    print('LMM price at K={}, {}'.format(KK, lmm.cal_price(KK, 'call')))
    # print(lmm.solve_bsm_vol([124, 125]))

    # bbg_df = pd.read_csv('data/AAPL_BBG_Volsurf_20200904.csv', index_col=0)
    # bbg_df = bbg_df[T_str] / 100

    lmm.plot(np.array(Ks), np.array(Prices), Callput)


def test_volsurface():
    lmms = Lognormal_mixture_model_surface(df_t_surf, forward, yc, valuation_date, p0)
    # lmms = Lognormal_mixture_model_surface_extended(df_t_surf, forward, yc, valuation_date, p0)
    # lmms.solve_param()

    return lmms


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


class Lognormal_mixture_model_surface_from_bsm_table(Lognormal_mixture_model_surface):
    def __init__(self, price_df, p0, Forward, yc, valuation_date, x0atm, N=4):
        lmms = []
        for t, forward in Forward._forward.items():
            t_year = (t - valuation_date).days / 365
            r = -np.log(yc.df(t)) / t_year
            if t in price_df.columns:
                print(t)
                lmm = Lognormal_mixture_model_curve(t_year, p0, forward, r, Forward._dividend_yield[t], N)
                if isinstance(x0atm, pd.Series):
                    x0_t = x0atm[t]
                    x0 = np.array([0.3, 0.3, 0.2, 0.2, 1, 1, 1, 0.75, x0_t, x0_t, x0_t, x0_t, 0])
                else:
                    x0 = np.array([0.3, 0.3, 0.2, 0.2, 1, 1, 1, 0.75, x0atm, x0atm, x0atm, x0atm, 0])
                callput = np.array(price_df.index.map(lambda x: 1 if x > Forward._forward[t] else -1))
                lmm.solve_param(np.array(price_df.index), np.array(price_df[t]), callput, x0)
                lmms.append([t, lmm])
        lmms = list(zip(*lmms))
        self.curves = pd.Series(lmms[1], index=lmms[0])
        self.listed_dates = lmms[0]
        self.valuation_date = valuation_date


def gen_model_from_table():
    price_table = '../data/IBM_BBG_Volsurf_20200817.csv'
    pt_raw = pd.read_csv(price_table, index_col=0)
    pt_raw /= 100
    # sample 5 points for each side
    Ks = np.array(pt_raw.index)
    k_anchor = Ks[np.argmin(np.abs(Ks - p0))]
    Ks_1 = np.random.choice(Ks[Ks < k_anchor], 5, replace=False)
    Ks_2 = np.random.choice(Ks[Ks > k_anchor], 5, replace=False)
    Ks_for_input = np.sort(Ks_1).tolist() + [k_anchor] + np.sort(Ks_2).tolist()
    pt = pt_raw.loc[Ks_for_input, :]
    pt.columns = pd.to_datetime(pt.columns)
    Ts = (pd.to_datetime(pt.columns) - valuation_date).days / 365
    for r in range(pt.shape[0]):
        for c in range(pt.shape[1]):
            rate = -np.log(yc.df(pt.columns[c])) / Ts[c]
            callput = 1 if pt.index[r] > Forward._forward[pt.columns[c]] else -1
            pt.iloc[r, c] = calc_bsm_price(rate, Forward._dividend_yield[pt.columns[c]], p0,
                                           pt.index[r], pt.iloc[r, c], Ts[c], callput=callput)
    lmms = Lognormal_mixture_model_surface_from_bsm_table(pt, p0, Forward, yc, valuation_date, x0atm)
    return lmms


ticker = 'V'
yc_file = 'data/V/mkt_yield_curve_20201002.xlsx'
quote_file = 'data/V_20201002.xlsx'
# forward_file = 'data/forward_AAPL_bbg_20200904.csv'
now = pd.to_datetime(datetime.date(2020, 10, 2))
# last = now - pd.offsets.BDay(1)
# df = get_option_data(ticker, last, now)
# div_df = get_proj_dividend(ticker)
# df = read_tick_quote_excel(quote_file, 'AMZN')
# df = df.drop('volume', axis=1)
df = read_mkt_from_excel(quote_file, 'V')
# df['implied_vol'] = df.implied_vol / 100  #for IBM

div_df, yc_df, p0 = get_batch_data_without_price(ticker, now, from_bbg=False)
valuation_date = now
T = df.expiry[~df.expiry.duplicated()].sort_values().reset_index(drop=True)
test_line = 11
T = T[test_line]
T_str = T.strftime('%Y-%m-%d')
yc = USD_yield_curve(yc_df, valuation_date)
# forw = pd.read_csv(forward_file, index_col=0, parse_dates=True)
# Forward = type('temp', (object,), {})()
# setattr(Forward, '_forward', forw['forward'])
# setattr(Forward, '_dividend_yield', forw['dividend_yield'] / 100)
# if hasattr(forw, 'atm_vol'):
#     x0atm = forw['atm_vol'] / 100

# div_df = get_proj_dividend(ticker)  # from BDVD
forward = Forward(div_df, df, p0, yc, valuation_date, american=True, from_hist_div=False)
f = forward.get_forward(T)
df = solve_aprox_atm_vol(df, yc, p0, forward, valuation_date)
df = data_filtering_for_quotes(df)

df_t = df[(df.expiry == T)].reset_index(drop=True)
# Forward._forward=Forward._forward[-3:]
df_t_surf = df.reset_index(drop=True)
# test_volcurve(df_t, T, forward.get_dividend_yield(T))

lmms = test_volsurface()
# test_volcurve(df_t, T, forward.get_dividend_yield(T), lmms, test_line)

# lmms.print_params()
output_matrix(ticker, lmms, forward, valuation_date, to_tenor=False)
output_matrix(ticker, lmms, forward, valuation_date, to_tenor=True)

# with pd.ExcelWriter('intertable_{}.xlsx'.format(ticker), datetime_format='yyyy-mm-dd') as ew:
#     inter.to_excel(ew)
# lmms = gen_model_from_table()
# given_T_forward = [124.69, 124.72, 123.45, 122.42, 121.38, 120.56]  # IBM 0817
# given_T_forward = [121.07, 121.09, 121.01, 120.98, 120.94, 120.97, 121.08, 121.19]  # AAPL 0904
# given_T_forward = [419.68, 419.84, 419.76, 420.36, 420.65, 420.71]  # TSLA 0904

# output_matrix(lmms, Forward, to_tenor=True, given_T_forward=given_T_forward)
