import QuantLib as ql
from scipy.optimize import brentq

CALL_PUT = {'call': 1, 'put': -1}


class _Vanilla_option:
    def __init__(self, yc, p0, dividend_yield, strike, callput, settlement_date, expiry_date, dt_valuation_date):
        p0, dividend_yield, strike = tuple(map(float, (p0, dividend_yield, strike)))
        self._expiry_date = ql.Date(expiry_date.strftime('%Y-%m-%d'), '%Y-%m-%d')
        self._settlement_date = ql.Date(settlement_date.strftime('%Y-%m-%d'), '%Y-%m-%d')
        self._valuation_date = valuation_date = ql.Date(dt_valuation_date.strftime('%Y-%m-%d'), '%Y-%m-%d')
        if not isinstance(callput, int) and callput.lower() in CALL_PUT.keys():
            callput = CALL_PUT[callput.lower()]
        callput = ql.Option.Call if callput == 1 else ql.Option.Put
        self._calendar = calendar = ql.UnitedStates()
        self._day_count = day_count = ql.Thirty360()
        ql.Settings.instance().evaluationDate = valuation_date
        self._payoff = ql.PlainVanillaPayoff(callput, strike)

        self._forward_handle = ql.QuoteHandle(ql.SimpleQuote(p0))
        self._dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, ql.QuoteHandle(
            ql.SimpleQuote(dividend_yield)), day_count, ql.Annual))
        self._yc_handle = ql.YieldTermStructureHandle(yc.get_curve())


class European_option(_Vanilla_option):
    def __init__(self, yc, p0, dividend_yield, strike, callput, settlement_date, expiry_date, valuation_date):
        super(European_option, self).__init__(yc, p0, dividend_yield, strike, callput, settlement_date, expiry_date,
                                              valuation_date)
        eu_exercise = ql.EuropeanExercise(self._expiry_date)
        self._european_option = ql.VanillaOption(self._payoff, eu_exercise)

    def __call__(self, vol):
        flat_vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(self._valuation_date, self._calendar, vol, self._day_count))
        bsm_process = ql.BlackScholesMertonProcess(self._forward_handle, self._dividend_handle, self._yc_handle,
                                                   flat_vol_ts)
        self._european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
        return self._european_option.NPV()


class American_option(_Vanilla_option):
    def __init__(self, yc, p0, dividend_yield, strike, callput, settlement_date, expiry_date, valuation_date):
        super(American_option, self).__init__(yc, p0, dividend_yield, strike, callput, settlement_date, expiry_date,
                                              valuation_date)
        am_excercise = ql.AmericanExercise(self._settlement_date, self._expiry_date)
        self._american_option = ql.VanillaOption(self._payoff, am_excercise)

    def __call__(self, vol, steps=200):
        flat_vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(self._valuation_date, self._calendar, vol, self._day_count))
        bsm_process = ql.BlackScholesMertonProcess(self._forward_handle, self._dividend_handle, self._yc_handle,
                                                   flat_vol_ts)
        # bsm_process = ql.BlackScholesProcess(self._forward_handle, self._yc_handle, flat_vol_ts)
        # binomial_engine = ql.BinomialVanillaEngine(bsm_process, 'crr', steps)
        binomial_engine = ql.FdBlackScholesVanillaEngine(bsm_process, steps, steps)
        self._american_option.setPricingEngine(binomial_engine)
        return self._american_option.NPV()


class American_solver:
    def __init__(self, yc, p0, dividend_yield, strike, callput, settlement_date, expiry_date, valuation_date,
                 cal_steps=200):
        self.option = American_option(yc, p0, dividend_yield, strike, callput, settlement_date, expiry_date,
                                      valuation_date)
        self.cal_steps = cal_steps

    def __call__(self, price):
        solve_func = lambda x: self.option(x, steps=self.cal_steps) - price
        return brentq(solve_func, 0.01, 5)


class European_solver:
    def __init__(self, yc, p0, dividend_yield, strike, callput, settlement_date, expiry_date, valuation_date):
        self.option = European_option(yc, p0, dividend_yield, strike, callput, settlement_date, expiry_date,
                                      valuation_date)

    def __call__(self, price):
        solve_func = lambda x: self.option(x) - price
        return brentq(solve_func, 0.01, 2)


if __name__ == '__main__':
    from src.USD_yield_curve import *
    from src.iostream import *

    ticker = 'IBM US'
    yc_file = 'data/mkt_yield_curve_20200817.xlsx'
    p0 = 124.495
    strike = 125
    callput = 1
    settlement_date = valuation_date = datetime.date(2020, 8, 17)
    expiry_date = datetime.date(2020, 8, 21)
    # vol = 29.403 / 100
    vol = 0.20523
    # div_df = get_proj_dividend(ticker)
    # div_df = div_df[div_df.date < expiry_date.strftime('%Y-%m-%d')]
    # if len(div_df):
    #     dividend_yield = div_df.dividend.cumsum().max() / (expiry_date - valuation_date).days * 365 / p0
    # else:
    #     dividend_yield = 0
    dividend_yield = 0 / 100
    yc = USD_yield_curve(read_mkt_yield_curve(yc_file), valuation_date)
    # yc=Constant_rate_curve(0.26475/100,valuation_date)
    american = American_option(yc, p0, dividend_yield, strike, callput, settlement_date, expiry_date, valuation_date)
    print(american(vol))

    solv = American_solver(yc, p0, dividend_yield, strike, callput, settlement_date, expiry_date, valuation_date)
    print(solv(0.83))

    eur = European_option(yc, p0, dividend_yield, strike, callput, settlement_date, expiry_date, valuation_date)
    print(eur(vol))
