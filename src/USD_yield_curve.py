import QuantLib as ql
import datetime
import numpy as np
import pandas as pd


class _yield_curve:
    def __init__(self):
        self.yield_curve = None

    def df(self, date):
        try:
            iter(date)
        except:
            date_str = date.strftime('%Y-%m-%d')
            return self.yield_curve.discount(ql.Date(date_str, '%Y-%m-%d'))
        else:
            date_ls = [self.yield_curve.discount(ql.Date(d.strftime('%Y-%m-%d'), '%Y-%m-%d')) for d in date]
            return np.array(date_ls)

    def get_curve(self):
        return self.yield_curve


class USD_yield_curve(_yield_curve):
    def __init__(self, df, valuation_date):
        calendar = ql.UnitedStates()
        df = df.append({'Maturity Date': valuation_date, 'Zero Rate': df['Zero Rate'][0]},
                       ignore_index=True)
        df = df.sort_values('Maturity Date')
        dates = df['Maturity Date'].map(lambda x: ql.Date(x.strftime('%Y-%m-%d'), '%Y-%m-%d')).to_list()
        zero_rate = (df['Zero Rate'] / 100).to_list()
        self.yield_curve = ql.ZeroCurve(dates, zero_rate, ql.Thirty360(), calendar)


class Constant_rate_curve(_yield_curve):
    def __init__(self, rate, valuation_date):
        self.yield_curve = ql.FlatForward(ql.Date(valuation_date.strftime('%Y-%m-%d'), '%Y-%m-%d'), rate,
                                          ql.Actual365Fixed())


class _USD_yield_curve(_yield_curve):
    def __init__(self):
        # market quotes
        # Update deposit Rates ( usual source will be LIBOR Fixings on the Curve Date
        deposits = {(1, ql.Weeks): 0.11063 / 100,
                    (1, ql.Months): 0.16688 / 100,
                    (2, ql.Months): 0.21888 / 100,
                    (3, ql.Months): 0.26825 / 100,
                    (6, ql.Months): 0.31750 / 100,
                    (12, ql.Months): 0.46050 / 100}
        # Obtain Futures prices from CME traded Euro Dollar Futures
        # futures = {ql.Date(17, 8, 2020): 99.7625,
        #            ql.Date(14, 9, 2020): 99.765,
        #            ql.Date(19, 10, 2020): 99.76,
        #            ql.Date(16, 11, 2020): 99.75,
        #            ql.Date(14, 12, 2020): 99.72,
        #            ql.Date(15, 3, 2021): 99.805,
        #            ql.Date(14, 6, 2021): 99.825,
        #            ql.Date(13, 9, 2021): 99.825}
        futures = {ql.Date(16, 9, 2020): 99.765,
                   ql.Date(16, 12, 2020): 99.72,
                   ql.Date(17, 3, 2021): 99.805,
                   ql.Date(16, 6, 2021): 99.825,
                   ql.Date(15, 9, 2021): 99.825}
        # Obtain ICE Swap rates from Traded Swaps on the Curve data
        swaps = {(1, ql.Years): 0.24400 / 100,
                 (2, ql.Years): 0.21500 / 100,
                 (3, ql.Years): 0.22200 / 100,
                 (4, ql.Years): 0.25300 / 100,
                 (5, ql.Years): 0.30400 / 100,
                 (6, ql.Years): 0.36600 / 100,
                 (7, ql.Years): 0.42800 / 100,
                 (8, ql.Years): 0.48400 / 100,
                 (9, ql.Years): 0.53600 / 100,
                 (10, ql.Years): 0.58000 / 100,
                 (15, ql.Years): 0.72600 / 100,
                 (20, ql.Years): 0.79800 / 100,
                 (30, ql.Years): 0.82300 / 100}
        # convert them to Quote objects
        for n, unit in deposits.keys():
            deposits[(n, unit)] = ql.SimpleQuote(deposits[(n, unit)])
        for d in futures.keys():
            futures[d] = ql.SimpleQuote(futures[d])
        for n, unit in swaps.keys():
            swaps[(n, unit)] = ql.SimpleQuote(swaps[(n, unit)])
        calendar = ql.UnitedStates()
        settlement_date = ql.Date(datetime.date(2020, 7, 29).strftime('%Y-%m-%d'), '%Y-%m-%d')
        ql.Settings.instance().evaluationDate = settlement_date

        # build rate helpers
        dayCounter = ql.Actual360()
        settlementDays = 2
        depositHelpers = [ql.DepositRateHelper(ql.QuoteHandle(deposits[(n, unit)]),
                                               ql.Period(n, unit), settlementDays,
                                               calendar, ql.ModifiedFollowing,
                                               False, dayCounter)
                          for n, unit in [(1, ql.Weeks), (1, ql.Months), (3, ql.Months),
                                          (6, ql.Months)]]

        dayCounter = ql.Actual360()
        months = 3
        futuresHelpers = [ql.FuturesRateHelper(ql.QuoteHandle(futures[d]),
                                               d, months,
                                               calendar, ql.ModifiedFollowing,
                                               False, dayCounter,
                                               ql.QuoteHandle(ql.SimpleQuote(0.0)))
                          for d in futures.keys()]

        settlementDays = 2
        fixedLegFrequency = ql.Semiannual
        fixedLegTenor = ql.Period(6, ql.Months)
        fixedLegAdjustment = ql.Unadjusted
        fixedLegDayCounter = ql.Thirty360()
        floatingLegFrequency = ql.Quarterly
        floatingLegTenor = ql.Period(3, ql.Months)
        floatingLegAdjustment = ql.ModifiedFollowing
        swapHelpers = [ql.SwapRateHelper(ql.QuoteHandle(swaps[(n, unit)]),
                                         ql.Period(n, unit), calendar,
                                         fixedLegFrequency, fixedLegAdjustment,
                                         fixedLegDayCounter, ql.USDLibor(ql.Period('3M')))
                       for n, unit in swaps.keys()]

        # discountTermStructure = ql.RelinkableYieldTermStructureHandle()
        # forecastTermStructure = ql.RelinkableYieldTermStructureHandle()
        helpers = depositHelpers[: 2] + futuresHelpers[:-1] + swapHelpers
        self.yield_curve = ql.PiecewiseSplineCubicDiscount(settlement_date, helpers,
                                                           ql.Actual360())


if __name__ == '__main__':
    import pandas as pd

    # import matplotlib.pyplot as plt
    # from iostream import read_mkt_yield_curve
    #
    # dates = pd.date_range("2020-08-15", "2030-08-01", freq="1m")
    # df = read_mkt_yield_curve('data/mkt_yield_curve.xlsx')
    valuation_date = pd.to_datetime('2020-08-14', format='%Y-%m-%d')
    # yc = USD_yield_curve(df, valuation_date)
    # dfs = [yc.df(d) for d in dates]
    #
    # plt.plot(dates, dfs)
    # plt.show()
    print(Constant_rate_curve(2.54 / 100, valuation_date).df(valuation_date + pd.to_timedelta('30d')))
