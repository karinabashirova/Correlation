import pandas as pd
from math import *
import numpy as np
import warnings
import os
from scipy import stats
import argparse
import plotly.graph_objects as go
import copy
import pwlf
from sklearn.linear_model import LinearRegression
from GPyOpt.methods import BayesianOptimization
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


class Price:
    def __init__(self, type):
        if type == 'ask_c' or type == 'bid_c' or type == 'mid_c':
            self.type = 'call'
        else:
            self.type = 'put'


class Volatility:
    def __init__(self, option_type):
        self.option_type = option_type

    def sigma_from_price(self, C, strike, expiry, r, error, S, is_forward=True):
        if is_forward:
            S = exp(-r * expiry) * S
        sigma = 1.
        sigma_old = 1.
        dv = error + 1.
        count = 0
        price_error_old, price_error_new = 20000, 10000
        while count <= 100 and abs(price_error_new) < abs(price_error_old):
            # while abs(dv) > error:
            price_error_old = price_error_new
            sigma_old = sigma
            count += 1
            d1 = (log(S / strike) + (r + 0.5 * sigma ** 2) * expiry) / (sigma * sqrt(expiry))
            d2 = d1 - sigma * sqrt(expiry)
            price = self.C_(d1, d2, strike, r, expiry, S)
            vega = self.Vega(S, d1, expiry)
            price_error_new = price - C
            dv = price_error_new / vega
            sigma = (sigma - dv)
            if abs(sigma) > 10.0:
                sigma = sigma_old
                break
        return sigma

    # def sigma_from_price(self, C, E, expiry, r, error, S):
    #     precision = 0.000001
    #     upper_vol = 500.0
    #     max_vol = 500.0
    #     lower_vol = 0.000001
    #     iteration = 0
    #
    #     while 1:
    #         iteration += 1
    #         mid_vol = (upper_vol + lower_vol) / 2.0
    #         d1 = (log(S / E) + (r + 0.5 * mid_vol ** 2) * expiry) / (mid_vol * sqrt(expiry))
    #         d2 = d1 - mid_vol * sqrt(expiry)
    #         price = self.C_(d1, d2, E, r, expiry, S)
    #         if self.option_type == 'call':
    #             d1 = (log(S / E) + (r + 0.5 * lower_vol ** 2) * expiry) / (lower_vol * sqrt(expiry))
    #             d2 = d1 - lower_vol * sqrt(expiry)
    #             lower_price = self.C_(d1, d2, E, r, expiry, S)
    #             if (lower_price - C) * (price - C) > 0:
    #                 lower_vol = mid_vol
    #             else:
    #                 upper_vol = mid_vol
    #             if abs(price - C) < precision: break
    #             if mid_vol > max_vol - 5:
    #                 mid_vol = 0.000001
    #                 break
    #
    #         elif self.option_type == 'put':
    #             d1 = (log(S / E) + (r + 0.5 * upper_vol ** 2) * expiry) / (upper_vol * sqrt(expiry))
    #             d2 = d1 - upper_vol * sqrt(expiry)
    #             upper_price = self.C_(d1, d2, E, r, expiry, S)
    #
    #             if (upper_price - C) * (price - C) > 0:
    #                 upper_vol = mid_vol
    #             else:
    #                 lower_vol = mid_vol
    #             if abs(price - C) < precision:
    #                 break
    #             if iteration > 50:
    #                 break
    #
    #     return mid_vol

    def N(self, x):
        return stats.norm.cdf(x)

    def Vega(self, S, d1, expiry):
        vega = S * stats.norm.pdf(d1) * sqrt(expiry)
        return vega

    def C_(self, d1, d2, strike, r, expiry, S):
        if self.option_type == 'call':
            return S * self.N(d1) - strike * exp(-r * expiry) * self.N(d2)
        else:
            return -S * self.N(-d1) + strike * exp(-r * expiry) * self.N(-d2)


class Reader:
    def __init__(self, type, path):
        self.df_options = pd.DataFrame()
        self.path = path
        self.strikes = []
        self.time = []
        self.spot = []
        self.forward = []
        pr = Price(type)
        self.option_type = pr.type
        self.price_type = type
        self.expiry_date = []
        self.price = {'call': [], 'put': []}
        self.forward_average = []
        self.file_count = []

    def get_data_from_file(self):
        for k, folder in enumerate(os.listdir(self.path)):
            check = False
            tmp_spot = []
            tmp_time = []
            tmp_call = []
            tmp_put = []
            self.file_count.append(len(os.listdir(self.path + "\\" + folder)))
            for f in os.listdir(self.path + "\\" + folder):
                df = pd.read_csv(self.path + "\\" + folder + "\\" + f)

                if not check:
                    self.strikes.append(df['k'].values.tolist())
                    self.expiry_date.append(df['e'][0])
                    check = True

                tmp_spot.append(df['s0'][0])

                today = pd.to_datetime(df['q'][0], format='%Y-%m-%d %H:%M:%S')
                diff = pd.to_datetime(df['e'][0], format='%Y-%m-%d %H:%M:%S') - today

                hours, remainder = divmod(diff.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                tmp_time.append(
                    diff.days / 365 + hours / (365 * 24) + minutes / (365 * 24 * 60) + seconds / (365 * 24 * 60 * 60))

                if self.price_type == 'mid_c' or self.price_type == 'mid_p':
                    tmp_call.append(
                        (np.array(df.loc[df['k'].isin(self.strikes[k])]['ask_c'].values.tolist()) +
                         np.array(df.loc[df['k'].isin(self.strikes[k])]['bid_c'].values.tolist())) / 2
                    )
                    tmp_put.append(
                        (np.array(df.loc[df['k'].isin(self.strikes[k])]['ask_p'].values.tolist()) +
                         np.array(df.loc[df['k'].isin(self.strikes[k])]['bid_p'].values.tolist())) / 2
                    )
                else:
                    tmp_call.append(df.loc[df['k'].isin(self.strikes[k])][self.price_type[:-1] + 'c'].values.tolist())
                    tmp_put.append(df.loc[df['k'].isin(self.strikes[k])][self.price_type[:-1] + 'p'].values.tolist())

            self.price['call'].append(np.array(tmp_call).T)
            self.price['put'].append(np.array(tmp_put).T)
            self.spot.append(tmp_spot)
            self.time.append(tmp_time)

    def count_vol(self, spot_forward, is_forward=True):
        v = copy.deepcopy(self.price[self.option_type])

        vol = Volatility(self.option_type)
        for k in range(len(self.price[self.option_type])):
            for i in range(len(self.price[self.option_type][k])):

                print('.', end='')

                for j in range(len(self.price[self.option_type][k][i])):
                    if not np.isnan(self.price[self.option_type][k][i][j]):
                        v[k][i][j] = vol.sigma_from_price(self.price[self.option_type][k][i][j], self.strikes[k][i],
                                                          self.time[k][j], 0., 0.001, spot_forward[k][j],
                                                          is_forward=is_forward)
                    else:
                        v[k][i][j] = np.nan

        print()

        return v

    def count_forward_price(self, v):
        self.forward = copy.deepcopy(self.price[self.option_type])
        for k in range(len(self.price[self.option_type])):
            for i in range(len(self.price[self.option_type][k])):
                print('.', end='')
                for j in range(len(self.price[self.option_type][k][i])):
                    if not np.isnan(self.price[self.option_type][k][i][j]):
                        self.forward[k][i][j] = \
                            (self.price['call'][k][i][j] - self.price['put'][k][i][j] + self.strikes[k][i])
                        #   + self.spot[k][j])
                    else:
                        self.forward[k][i][j] = np.nan

        delta = copy.deepcopy(self.price[self.option_type])
        for k in range(len(self.price[self.option_type])):
            for i in range(len(self.price[self.option_type][k])):
                for j in range(self.file_count[k]):
                    delta[k][i][j] = self.delta(self.forward[k][i][j], self.strikes[k][i], self.time[k][j], v[k][i][j])

        for k in range(len(self.price[self.option_type])):
            delta[k] = np.array(delta[k])
            mean = []
            for j in range(self.file_count[k]):
                delta_sum = 0
                for i in range(len(self.price[self.option_type][k])):
                    if not np.isnan(delta[k][i][j]):
                        delta_sum += delta[k][i][j]

                tmp = []
                for i in range(len(self.price[self.option_type][k])):
                    if not np.isnan(delta[k][i][j]) and not np.isnan(self.forward[k][i][j]):
                        tmp.append(self.forward[k][i][j] * delta[k][i][j] / delta_sum)

                if (np.nansum(tmp)) != 0:
                    mean.append(np.nansum(tmp))
                else:
                    mean.append(np.nan)

            self.forward_average.append(mean)

        print()

    def delta(self, forward, strike, time, volatility):
        if self.option_type == 'call':
            return self.N(self.d1(forward, strike, time, volatility))
        else:
            return self.N(self.d1(forward, strike, time, volatility)) - 1

    def d1(self, forward, strike, time, volatility):
        return (log(forward / strike) + (0.5 * volatility ** 2) * time) / (volatility * sqrt(time))

    def N(self, x):
        return stats.norm.cdf(x)

    def plot_forward(self):
        for k in range(len(self.price[self.option_type])):
            fig = go.Figure()

            for i in range(len(self.price[self.option_type][k])):
                fig.add_trace(
                    go.Scatter(x=self.time[k], y=self.forward[k][i],  # / self.strikes[k][i] - 1, #  (self.spot[k])
                               mode="markers", marker=dict(size=4), name=self.strikes[k][i]))
            fig.add_trace(
                go.Scatter(x=self.time[k], y=self.forward_average[k],  # / self.strikes[k][i] - 1,  # (self.spot[k])
                           mode="markers+lines", marker=dict(size=6), name='forward average'))
            fig.add_trace(
                go.Scatter(x=self.time[k], y=self.spot[k],  # / self.strikes[k][i] - 1,  # (self.spot[k])
                           mode="markers+lines", marker=dict(size=6), name='spot'))

            fig.update_xaxes(title_text="Time to expiration")
            fig.update_yaxes(title_text="Forward")

            fig.update_layout(title_text="Forward for expiration time " + str(self.expiry_date[k]) +
                                         ' for ' + str(self.price_type))

            fig.show()

    def lin_reg(self, X, y):
        reg = LinearRegression().fit(X.reshape(len(X), -1), y)
        pred = reg.predict(X.reshape(len(X), -1))
        return pred

    def plot_forward_rate(self):
        for k in range(len(self.price[self.option_type])):
            fig = go.Figure()

            x = np.array(self.time[k])
            y = np.array(self.forward_average[k]) / np.array(self.spot[k]) - 1
            indices = [not np.isnan(element) for element in self.forward_average[k]]
            x = x[indices]
            y = y[indices]

            x = np.array(x)

            my_pwlf = pwlf.PiecewiseLinFit(x, y)

            def my_obj(x):
                l = y.mean() * 0.001
                f = np.zeros(x.shape[0])
                for i, j in enumerate(x):
                    my_pwlf.fit(j[0])
                    f[i] = my_pwlf.ssr + (l * j[0])
                return f

            # define the lower and upper bound for the number of line segments
            max_count = int(self.time[k][0]*365)
            print(max_count)
            bounds = [{'name': 'var_1', 'type': 'discrete',
                       'domain': np.arange(2, max_count)}]

            np.random.seed(12121)

            myBopt = BayesianOptimization(my_obj, domain=bounds, model_type='GP',
                                          initial_design_numdata=int(len(x)/30),
                                          initial_design_type='random',
                                          exact_feval=True, verbosity=True,
                                          verbosity_model=False)
            max_iter = 30

            myBopt.run_optimization(max_iter=max_iter, verbosity=True)

            my_pwlf.fit(myBopt.x_opt)
            xHat = np.linspace(min(x), max(x), num=len(x))

            print('len xHat', len(xHat))

            xHat = xHat[::-1]
            yHat = [my_pwlf.predict(xHat[0])[0]]
            for i in range(1, len(xHat)):
                y_ = my_pwlf.predict(xHat[i])[0]

                if y_ <= yHat[i - 1]:
                    yHat.append(y_)
                else:
                    yHat.append(yHat[i - 1])

            xHat = xHat[::-1]
            yHat = yHat[::-1]
            fig.add_trace(
                go.Scatter(x=self.time[k], y=np.array(self.forward_average[k]) / np.array(self.spot[k]) - 1,
                           # (self.spot[k])
                           mode="lines+markers", marker=dict(size=4), name='average'))
            # fig.add_trace(
            #     go.Scatter(x=x_new, y=y_new,  # (self.spot[k])
            #                mode="lines+markers", marker=dict(size=4), name='interp1d'))
            # fig.add_trace(
            #     go.Scatter(x=x, y=p(x),  # (self.spot[k])
            #                mode="lines+markers", marker=dict(size=4), name='polyfit'))
            # fig.add_trace(
            #     go.Scatter(x=x, y=w,  # (self.spot[k])
            #                mode="lines+markers", marker=dict(size=4), name='savgol_filter'))
            # fig.add_trace(
            #     go.Scatter(x=x, y=y_pwlf,  # (self.spot[k])
            #                mode="lines+markers", marker=dict(size=4), name='pwlf'))
            # fig.add_trace(
            #     go.Scatter(x=x, y=ys_sl,  # (self.spot[k])
            #                mode="lines+markers", marker=dict(size=4), name='Tree'))
            fig.add_trace(
                go.Scatter(x=xHat, y=yHat,
                           mode="lines+markers", marker=dict(size=4), name='Optimization'))
            # fig.add_trace(
            #     go.Scatter(x=xHat, y=yHat1,
            #                mode="lines+markers", marker=dict(size=4), name='Before'))

            fig.update_xaxes(title_text="Time to expiration")
            fig.update_yaxes(title_text="Forward rate")

            fig.update_layout(title_text="Forward rate for expiration time " + str(self.expiry_date[k]) +
                                         ' for ' + str(self.price_type))

            fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Correlation between volatility and underlying price')
    parser.add_argument("t", type=str, choices=['ask_c', 'bid_c', 'mid_c', 'ask_p', 'bid_p', 'mid_p'],
                        help="Surface for ask_c/bid_c/mid_c")
    parser.add_argument('p', help='Path to files')

    args = parser.parse_args()

    reader = Reader(args.t, args.p)
    reader.get_data_from_file()

    v_spot = reader.count_vol(reader.spot, is_forward=False)
    reader.count_forward_price(v_spot)

    # reader.plot_forward()
    reader.plot_forward_rate()

    # volatility = reader.count_vol(reader.forward_average)

    # fig = []
    # for k in range(len(volatility)):
    #     df = pd.DataFrame()
    #     df['Strike ' + str(k)] = reader.strikes[k]
    #
    #     tmp_corr = []
    #     for i in range(len(volatility[k])):
    #         # tmp = [(x, y) for x, y in zip(reader.forward[k][i], volatility[k][i]) if not np.isnan(y)]
    #         tmp = [(x, y) for x, y in zip(reader.forward_average[k], volatility[k][i]) if not np.isnan(y)]
    #         # tmp = [(x, y) for x, y in zip(reader.price['call'][k][i], volatility[k][i]) if not np.isnan(y)]
    #         tmp_spot = [tmp[i][0] for i in range(len(tmp))]
    #         tmp_volatility = [tmp[i][1] for i in range(len(tmp))]
    #
    #         if len(tmp_volatility) != 0:
    #             tmp_corr.append(np.corrcoef(tmp_spot, tmp_volatility)[0, 1])
    #         else:
    #             tmp_corr.append(np.nan)
    #     tmp_corr = np.array(tmp_corr)
    #
    #     df['Correlation coefficient ' + str(k)] = tmp_corr
    #     df.to_csv('forward_correlation' + reader.expiry_date[k].split()[0] + '.csv', index=False)
    #
    #     tmp = tmp_corr[np.logical_not(np.isnan(tmp_corr))]
    #     indexes = np.arange(len(tmp_corr))
    #     indexes = indexes[np.logical_not(np.isnan(tmp_corr))]
    #
    #     fig.append(make_subplots(rows=len(tmp), cols=1, start_cell="bottom-left",
    #                              subplot_titles=[
    #                                  'r = ' + str(round(t, 3)) + ' for strike ' + str(reader.strikes[k][i])
    #                                  for i, t in zip(indexes, tmp)]))
    #
    #     for i, index in zip(np.arange(len(indexes)), indexes):
    #         fig[k].add_trace(
    #             # go.Scatter(x=reader.forward[k][index], y=volatility[k][index], mode="markers",  # reader.spot[k]
    #             go.Scatter(x=reader.forward_average[k], y=volatility[k][index], mode="markers",
    #             # go.Scatter(x=reader.price['call'][k][index], y=volatility[k][index], mode="markers",
    #                                   # reader.spot[k]
    #                                   marker=dict(color=reader.time[k])), row=i + 1, col=1)
    #
    #         # fig[k].add_trace(
    #         #     go.Scatter(x=reader.time[k], y=volatility[k][index], mode="lines",  # reader.spot[k]
    #         #                marker=dict(color=reader.time[k])), row=i + 1, col=1)
    #
    #
    #         fig[k].update_xaxes(title_text="Forward price", row=i + 1, col=1)
    #         fig[k].update_yaxes(title_text="Volatility", row=i + 1, col=1)
    #
    #         fig[k].update_layout(showlegend=False)
    #         fig[k].update_layout(height=len(tmp) * 250, width=600,
    #                              title_text="Correlation for expiration time " + str(
    #                                  reader.expiry_date[k]) + '\nfor ' + str(reader.price_type))
    #
    # for f in fig:
    #     f.show()
