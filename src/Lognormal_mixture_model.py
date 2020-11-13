from collections.abc import Iterable

import scipy.optimize as opt
from scipy.interpolate import PchipInterpolator
from scipy.sparse import dia_matrix

from src.Forward import _de_americanize_vol
from src.utils import *

CALL_PUT = {'call': 1, 'put': -1}


class Lognormal_mixture_model_curve:
    def __init__(self, T, spot, forward, r, d, valuation_date, N=4):
        self.p = np.zeros((N,))
        self.xi = np.zeros((N,))
        self.sigma = np.zeros((N,))
        self.T = T
        self.spot = spot
        self.forward = forward
        self.r = r
        self.d = d
        self.N = N
        self.shape = (N,)
        self.x_shape = (3 * N,)
        self.valuation_date = valuation_date
        self.atm_vol = 0

    # for creating object with manually-set parameters
    def set_param(self, p, xi, sigma):
        if not p.shape == xi.shape == sigma.shape == self.shape:
            raise ValueError('The length of Ks is not equal to the modal.')
        self.p = p
        self.xi = xi
        self.sigma = sigma
        return self

    def _ds(self, Ks):
        d1 = (np.log(self.xi * self.spot / Ks) + (self.r - self.d + 0.5 * np.square(self.sigma)) * self.T) / (
                self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    # parse x0 to the object
    def _parse_param(self, x):
        x = x[:-1].reshape((3, self.shape[0]))
        self.p = x[0, :]
        self.xi = x[1, :]
        self.sigma = x[2, :]
        return self

    # calculate LNM Black and Scholes price
    def cal_price(self, Ks, callput):
        Ks = np.array(Ks).reshape((-1, 1))
        callput = np.array(callput).reshape((-1, 1))
        assert Ks.shape == callput.shape
        try:
            int(callput[0])
        except:
            callput = np.vectorize(lambda x: CALL_PUT[x.lower()])(callput)
        d1, d2 = self._ds(Ks)
        price = np.sum(
            self.p * (self.xi * self.spot * np.exp(-self.d * self.T) * norm.cdf(callput * d1) - Ks * np.exp(
                -self.r * self.T) * norm.cdf(callput * d2)) * callput, axis=1)
        return price

    # df/dpi(function to option price)
    def _dfdpi(self, Ks, Ps, callput):
        return 2 * (self.cal_price(Ks, callput).reshape((-1, 1)) - Ps)

    # df/dp
    def _dfdp(self, Ks, Ps, callput):
        dfdpi = self._dfdpi(Ks, Ps, callput)
        d1, d2 = self._ds(Ks)
        dpidp = (self.xi * self.spot * np.exp(-self.d * self.T) * norm.cdf(callput * d1) - Ks * np.exp(
            -self.r * self.T) * norm.cdf(callput * d2)) * callput
        return dfdpi.T.dot(dpidp)

    # df/dxi (delta)
    def _dfdxi(self, Ks, Ps, callput):
        dfdpi = self._dfdpi(Ks, Ps, callput)
        d1, _ = self._ds(Ks)
        dpidxi = self.p * self.spot * np.exp(-self.d * self.T) * norm.cdf(callput * d1) * callput
        return dfdpi.T.dot(dpidxi)

    # df/dsigma (vega)
    def _dfdsigma(self, Ks, Ps, callput):
        dfdpi = self._dfdpi(Ks, Ps, callput)
        _, d2 = self._ds(Ks)
        dpidsigma = self.p * Ks * np.exp(-self.r * self.T) * norm.pdf(d2) * np.sqrt(self.T)
        return dfdpi.T.dot(dpidsigma)

    # calculate the Jacobians of each param in the LNM model
    def _jac(self, x, Ks, Ps, callput):
        Ks = Ks.reshape((-1, 1))
        Ps = Ps.reshape((-1, 1))
        callput = callput.reshape((-1, 1))
        dfdp = self._dfdp(Ks, Ps, callput)
        dfdxi = self._dfdxi(Ks, Ps, callput)
        dfdsigma = self._dfdsigma(Ks, Ps, callput)
        q = np.zeros((dfdp.shape[0], 1))
        J = np.concatenate([dfdp, dfdxi, dfdsigma, q], axis=1)
        return J

    # calculate the Jacobians of each param in the atm vol constraint
    def _jac_atm(self, x, k, p, callput):
        d1, d2 = self._ds(k)
        dfdp = (self.xi * self.spot * np.exp(-self.d * self.T) * norm.cdf(callput * d1) - k * np.exp(
            -self.r * self.T) * norm.cdf(callput * d2)) * callput
        dfdxi = self.p * self.spot * np.exp(-self.d * self.T) * norm.cdf(callput * d1) * callput
        dfdsigma = self.p * k * np.exp(-self.r * self.T) * norm.pdf(d2) * np.sqrt(self.T)
        J = np.concatenate([dfdp, dfdxi, dfdsigma, [0]], axis=0)
        return J

    # solve LNM parameters
    def solve_param(self, Ks, Ps, callput, x0, lmm_last, anchor_atm=False, k=None, k_vol=None):
        if not Ks.shape == Ps.shape == callput.shape:
            raise ValueError('The length of Ks/Ps is not equal to the Ps.')
        if isinstance(callput[0], str):
            callput = np.vectorize(lambda x: CALL_PUT[x.lower()])(callput)
        scale = Ps.mean() * np.sqrt(len(Ps))
        func = lambda x: np.sum(np.square((self._parse_param(x).cal_price(Ks, callput) - Ps) / scale))
        jac = lambda x: self._jac(x, Ks, Ps, callput) / scale / scale
        N = self.shape[0]
        if lmm_last != 0:
            t_last = (lmm_last[0] - self.valuation_date).days / 365
            sigma_lb = lmm_last[1].sigma * np.sqrt(t_last) / np.sqrt(self.T)
            lb = np.array([1e-5] * N * 2 + sigma_lb.tolist() + [max(1 - sum(lmm_last[1].p), 0)])
        else:
            lb = np.array([1e-5] * N * 2 + [0.01] * N + [0])
        ub = np.array([1] * N + [np.inf] * N + [np.inf] * N + [1])
        bounds = opt.Bounds(lb, ub)
        # x0 = trim_sigma(x0, lb, self.N)
        if anchor_atm:
            cp_atm = 1 if k >= self.forward else -1
            p_atm = calc_bsm_price(self.r, self.d, self.spot, k, k_vol, self.T, cp_atm)
            constr_func = lambda x: np.array([np.sum(x[:N]) + x[-1] - 1,
                                              x[:N].dot(x[N:2 * N]) - 1,
                                              np.squeeze(self.cal_price(k, cp_atm) - p_atm)])
            constr_jac = lambda x: np.array([[1] * N + [0] * N * 2 + [1],
                                             np.concatenate([x[N:2 * N], x[:N], [0] * N, [0]]),
                                             self._jac_atm(x, k, p_atm, cp_atm).tolist()])
        else:
            constr_func = lambda x: np.array([np.sum(x[:N]) + x[-1] - 1,
                                              x[:N].dot(x[N:2 * N]) - 1])
            constr_jac = lambda x: np.array([[1] * N + [0] * N * 2 + [1],
                                             np.concatenate([x[N:2 * N], x[:N], [0] * N, [0]])])
        eq_cons = {'type': 'eq',
                   'fun': constr_func,
                   'jac': constr_jac}
        reg = opt.minimize(func, x0, method='SLSQP', jac=jac, constraints=[eq_cons], options={'disp': True,
                                                                                              'ftol': 1e-11,
                                                                                              'maxiter': 1000},
                           bounds=bounds)
        self._parse_param(reg.x)
        Sigmas = self.sigma * np.sqrt(self.T)
        d1 = np.log(self.xi) / Sigmas + 0.5 * Sigmas
        d2 = np.log(self.xi) / Sigmas - 0.5 * Sigmas
        self.atm_vol = 2 / np.sqrt(self.T) * norm.ppf(
            0.5 + 0.5 * np.sum(self.p * (self.xi * norm.cdf(d1) - norm.cdf(d2))))
        self._sort_params()
        return reg

    # find initial x0 with unconstraint method
    def find_x0(self, x0, Ks, Ps, callput):
        func = lambda x: self._parse_param(x).cal_price(Ks, callput) - Ps
        reg = opt.root(func, x0, method='lm', options={'maxiter': 5000})
        x0 = trim_x(reg.x, self.N)
        return x0

    # globally find optimized parameters
    def solve_global(self, Ks, Ps, callput):
        func = lambda x: np.sum(np.square(self._parse_param(x).cal_price(Ks, callput) - Ps))
        jac = lambda x: self._jac(x, Ks, Ps, callput)
        N = self.shape[0]
        lb = np.array([1e-5] * N * 2 + [0.01] * N + [0])
        ub = np.array([1] * N + [3] * N + [3] * N + [1])
        bounds = list(zip(lb, ub))
        constr_func = lambda x: np.array([np.sum(x[:N]) + x[-1] - 1,
                                          x[:N].dot(x[N:2 * N]) - 1])
        constr_jac = lambda x: np.array([[1] * N + [0] * N * 2 + [1],
                                         np.concatenate([x[N:2 * N], x[:N], [0] * N, [0]])])
        eq_cons = {'type': 'eq',
                   'fun': constr_func,
                   'jac': constr_jac}
        reg = opt.shgo(func, bounds=bounds, constraints=[eq_cons], minimizer_kwargs={'method': 'SLSQP'},
                       options={'jac': jac, 'maxtime': 10})
        self._parse_param(reg.x)
        return reg

    @staticmethod
    def _bsm_fun(x, m, y):
        return norm.cdf(m / x + 0.5 * x) - np.exp(-m) * norm.cdf(m / x - 0.5 * x) - y

    @staticmethod
    def _bsm_jac(x, m, y):
        return np.diag(norm.pdf(m / x + 0.5 * x) * (-m / np.square(x) + 0.5) - np.exp(-m) * norm.pdf(
            m / x - 0.5 * x) * (-m / np.square(x) - 0.5))

    # solve for Black and Scholes local volatility
    def solve_bsm_vol(self, Ks):
        Ks = np.array(Ks).reshape((-1, 1))
        m = np.log(self.forward / Ks) + self.r * self.T
        m_xi = np.log(self.forward * self.xi / Ks) + self.r * self.T
        d1 = m_xi / self.sigma / np.sqrt(self.T) + 0.5 * self.sigma * np.sqrt(self.T)
        d2 = d1 - self.sigma * np.sqrt(self.T)
        Ps = np.sum(self.xi * self.p * (norm.cdf(d1) - np.exp(-m_xi) * norm.cdf(d2)), axis=1)
        m = m.flatten()
        x0 = np.array([0.2] * len(Ks))
        reg = opt.least_squares(self._bsm_fun, x0, jac=self._bsm_jac, jac_sparsity=dia_matrix(np.diag([1] * len(m))),
                                bounds=(0, 2),
                                args=(m, Ps)
                                )
        return reg.x / np.sqrt(self.T)

    # sort N parameters in terms of ascending volatility
    def _sort_params(self):
        params = list(zip(self.p, self.xi, self.sigma))
        params.sort(key=lambda x: x[-1])
        params_new = np.array(list(zip(*params)))
        self.set_param(p=params_new[0, :], xi=params_new[1, :], sigma=params_new[2, :])

    # plot curve
    def plot(self, Ks=None, Ps=None, callput=None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import MultiCursor
        fig, ax = plt.subplots(1, 2)
        k_x = np.linspace(self.spot * 0.5, self.spot * 1.5, 200)
        callput_x = np.vectorize(lambda x: 1 if x > self.forward else -1)(k_x)
        ax[0].plot(k_x, self.cal_price(k_x, callput_x))
        if Ks is not None:
            KC = list(zip(Ks, callput, Ps))
            KC.sort(key=lambda x: x[0])
            Ks, Callput, Prices = tuple(zip(*KC))
            ax[0].plot(Ks, Prices, 'r.')
        ax[0].set_title('Fit Market Quote')

        ax[1].plot(k_x, self.solve_bsm_vol(k_x), label='Python solver')
        ax[1].set_title('Volatility Curve Comparison')
        ax[1].legend()
        cursor = MultiCursor(fig.canvas, (ax[0], ax[1]), horizOn=True, vertOn=True, useblit=True, color='black', lw=1,
                             ls=':')
        plt.show()


class Lognormal_mixture_model_surface:
    def __init__(self, price_df, forward, yc, valuation_date, p0, ticker, atmguess=False, N=4):
        # input dataframe: quote price table
        lmms = []
        Ts = price_df.expiry[~price_df.expiry.duplicated()].sort_values().reset_index(drop=True)
        for t in Ts:
            forw = forward.get_forward(t)
            t_year = (t - valuation_date).days / 365
            r = -np.log(yc.df(t)) / t_year
            subprice = price_df[price_df.expiry == t].reset_index(drop=True).copy()
            if len(subprice) > 0:
                print(t)
                option_mask = ((subprice.putcall == 'call') & (subprice.strike >= forw / yc.df(t))) | (
                        (subprice.putcall == 'put') & (subprice.strike < forw / yc.df(t)))
                subprice = subprice[option_mask].reset_index(drop=True)
                k_t = np.argmin(np.abs(subprice.strike - p0))
                k_t = subprice.strike[k_t]
                atm_price = subprice[subprice.strike == k_t].reset_index().iloc[0]
                x0_t = atm_price.implied_vol
                anchor = True if atmguess else False
                for row in range(subprice.shape[0]):
                    subprice.loc[row, 'price'] = _de_americanize_vol(subprice.loc[row, 'price'], yc, p0,
                                                                     forward.get_dividend_yield(t),
                                                                     subprice.loc[row, 'strike'],
                                                                     subprice.loc[row, 'putcall'],
                                                                     valuation_date, subprice.loc[row, 'expiry'])
                subprice = calculate_mid_price(subprice)
                lmm = Lognormal_mixture_model_curve(t_year, p0, forw, r, forward.get_dividend_yield(t),
                                                    valuation_date,
                                                    N)
                x0 = np.array(
                    [0.4, 0.3, 0.2, 0.1 - 1e-7, 1, 0.8, 1.3, 1. + 1.001e-6, x0_t, x0_t, x0_t, x0_t, 1e-7]),
                if len(lmms):
                    lmm.solve_param(np.array(subprice.strike),
                                    np.array(subprice.price),
                                    np.array(subprice.putcall),
                                    x0,
                                    lmms[-1],
                                    anchor,
                                    k_t,
                                    x0_t)
                else:
                    lmm.solve_param(np.array(subprice.strike),
                                    np.array(subprice.price),
                                    np.array(subprice.putcall),
                                    x0,
                                    0,
                                    anchor,
                                    k_t,
                                    x0_t)
                lmms.append([t, lmm, subprice])

        lmms = list(zip(*lmms))
        # cache the prices that have been pre-processed
        reduce(lambda x, y: pd.concat([x, y], axis=0), lmms[2]).to_csv('cached_data/{}_processed.csv'.format(ticker))
        self.curves = pd.Series(lmms[1], index=lmms[0])
        self.listed_dates = lmms[0]
        self.valuation_date = valuation_date
        self.forward = forward
        self.N = N

    def solve_vol_from_strike(self, K, Ts):
        # if isinstance(Ts, Iterable) and not isinstance(Ts, str):
        #     assert (max(Ts) <= self.listed_dates[-1]) and (min(Ts) >= self.listed_dates[0])  # Guarantee interpolation
        if isinstance(Ts, str):
            Ts = pd.to_datetime(Ts)
        Ts = pd.Series(Ts)
        Vol_points = np.vectorize(lambda x: x.solve_bsm_vol(K))(self.curves)
        Ts = np.vectorize(lambda x: (x - self.valuation_date).days / 365)(Ts)
        lds = np.vectorize(lambda x: (x - self.valuation_date).days / 365)(self.listed_dates)
        variance = np.square(Vol_points) * lds
        interp_var = PchipInterpolator(lds, variance)(Ts) / Ts
        return np.sqrt(interp_var)

    def solve_vol_from_forward_moneyness(self, m, Ts):
        # if isinstance(Ts, Iterable) and not isinstance(Ts, str):
        #     assert (max(Ts) <= self.listed_dates[-1]) and (min(Ts) >= self.listed_dates[0])  # Guarantee interpolation
        if isinstance(Ts, str):
            Ts = pd.to_datetime(Ts)
        if Ts > self.listed_dates[-1]:
            last_lmn = self.curves[-1]
            T_y = (Ts - self.valuation_date).days / 365
            new_lmn = Lognormal_mixture_model_curve(T_y, last_lmn.spot, self.forward.get_forward(Ts), last_lmn.r,
                                                    last_lmn.d,
                                                    self.valuation_date, self.N)
            new_lmn.set_param(p=last_lmn.p, xi=last_lmn.xi, sigma=last_lmn.sigma)
            return new_lmn.solve_bsm_vol(new_lmn.forward * m)
        else:
            Ts = pd.Series(Ts)
            Vol_points = np.vectorize(lambda x: x.solve_bsm_vol(x.forward * m))(self.curves)
            Ts = np.vectorize(lambda x: (x - self.valuation_date).days / 365)(Ts)
            lds = np.vectorize(lambda x: (x - self.valuation_date).days / 365)(self.listed_dates)
            variance = np.square(Vol_points) * lds
            interp_var = PchipInterpolator(lds, variance)(Ts) / Ts
            return np.sqrt(interp_var)

    def print_params(self):
        params = np.array([np.concatenate([x.p, x.xi, x.sigma, [1 - sum(x.p)]]) for x in self.curves])
        return params

    def __check_sigma(self):
        Ts = np.sqrt(np.array([(x - self.valuation_date).days / 365 for x in self.listed_dates])).reshape((-1, 1))
        sigmas = np.array([c.sigma for c in self.curves]) * Ts
        need_modify = False
        for c in range(sigmas.shape[1]):
            mod_list = []
            for r in range(1, sigmas.shape[0]):
                if max(sigmas[:r + 1, c]) != sigmas[r, c]:
                    mod_list.append(r - 1)
            if len(mod_list):
                need_modify = True
                for mod_num in mod_list:
                    mod_prev = mod_num - 1
                    while True:
                        if mod_prev not in mod_list:
                            break
                        else:
                            mod_prev -= 1
                    mod_next = mod_num + 1
                    while True:
                        if mod_next not in mod_list:
                            break
                        else:
                            mod_next += 1
                    if mod_prev == -1:
                        mod_prev = mod_next
                    sigmas[mod_num, c] = np.mean([sigmas[mod_prev, c], sigmas[mod_next, c]])
        if need_modify:
            sigmas /= Ts
            row = 0
            for curve in self.curves:
                curve.sigma = sigmas[row, :]
                row += 1
