import pandas as pd
from math import *
import numpy as np
import warnings
import os
from scipy import stats
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy
import matplotlib.pyplot as plt
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

    def sigma_from_price(self, C, strike, expiry, r, error, S):
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
                # print(sigma, price_error_new, dv, vega, d1, d2)
                break
            # if np.isnan(sigma):
            #     print('d1', d1, 'd2', d2)
            #     print(S, strike, expiry, dv, count, price_error_new)
            #     break
        # print('sigma', sigma)
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
        self.v = []
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
                        (np.array(df.loc[df['k'].isin(self.strikes[k])]['ask_p'].values.tolist()) -
                         np.array(df.loc[df['k'].isin(self.strikes[k])]['bid_p'].values.tolist())) / 2
                    )
                else:
                    tmp_call.append(df.loc[df['k'].isin(self.strikes[k])][self.price_type[:-1] + 'c'].values.tolist())
                    tmp_put.append(df.loc[df['k'].isin(self.strikes[k])][self.price_type[:-1] + 'p'].values.tolist())

            self.price['call'].append(np.array(tmp_call).T)
            self.price['put'].append(np.array(tmp_put).T)
            self.spot.append(tmp_spot)
            self.time.append(tmp_time)

    def count_vol(self):
        self.v = copy.deepcopy(self.price[self.option_type])

        vol = Volatility(self.option_type)
        for k in range(len(self.price[self.option_type])):
            for i in range(len(self.price[self.option_type][k])):

                print('.', end='')

                for j in range(len(self.price[self.option_type][k][i])):
                    if not np.isnan(self.price[self.option_type][k][i][j]):
                        self.v[k][i][j] = vol.sigma_from_price(self.price[self.option_type][k][i][j], self.strikes[k][i],
                                                          self.time[k][j], 0., 0.001, self.forward_average[k][j])
                                                                                         # self.spot[k][j])
                    else:
                        self.v[k][i][j] = np.nan
            print('v[k]', np.isnan(self.v[k]).sum())
        print()

    def count_forward_price(self, r, D):
        self.forward = copy.deepcopy(self.price[self.option_type])
        for k in range(len(self.price[self.option_type])):
            for i in range(len(self.price[self.option_type][k])):
                print('.', end='')
                for j in range(len(self.price[self.option_type][k][i])):
                    if not np.isnan(self.price[self.option_type][k][i][j]):
                        self.forward[k][i][j] = \
                            (-self.price['call'][k][i][j] + self.price['put'][k][i][j]  # + self.strikes[k][i])
                                + self.spot[k][j])
                                                                            # / (self.strikes[k][i]) - 1
                    else:
                        self.forward[k][i][j] = np.nan

        for k in range(len(self.price[self.option_type])):
            mean = []
            for j in range(self.file_count[k]):
                tmp = []
                for i in range(len(self.price[self.option_type][k])):
                    tmp.append(self.forward[k][i][j])
                mean.append(np.nanmean(tmp))

            self.forward_average.append(mean)

        print()

    def plot_forward(self):
        for k in range(len(self.price[self.option_type])):
            fig = go.Figure()

            for i in range(len(self.price[self.option_type][k])):
                fig.add_trace(
                    go.Scatter(x=self.time[k], y=self.forward[k][i],  # / self.strikes[k][i] - 1, #  (self.spot[k])
                               mode="markers", marker=dict(size=4), name=self.strikes[k][i]))
            fig.add_trace(
                go.Scatter(x=self.time[k], y=self.forward_average[k],  # / self.strikes[k][i] - 1,  # (self.spot[k])
                           mode="markers+lines", marker=dict(size=10), name='forward average'))
            fig.add_trace(
                go.Scatter(x=self.time[k], y=self.spot[k],  # / self.strikes[k][i] - 1,  # (self.spot[k])
                           mode="markers+lines", marker=dict(size=10), name='spot'))

            fig.update_xaxes(title_text="Time to expiration")
            fig.update_yaxes(title_text="Forward")

            fig.update_layout(title_text="Forward for expiration time " + str(self.expiry_date[k]) +
                                         ' for ' + str(self.price_type))

            fig.show()

    def plot_forward_rate(self):
        for k in range(len(self.price[self.option_type])):
            fig = go.Figure()

            for i in range(len(self.price[self.option_type][k])):
                fig.add_trace(
                    go.Scatter(x=self.time[k], y=self.forward[k][i] / self.strikes[k][i] - 1, #  (self.spot[k])
                               mode="lines+markers", marker=dict(size=4), name=self.strikes[k][i]))

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

    reader.count_forward_price(0.01, 1.)

    # reader.plot_forward()
    reader.plot_forward_rate()
    reader.count_vol()
    volatility = reader.v
    fig = []
    for k in range(len(volatility)):
        df = pd.DataFrame()
        df['Strike ' + str(k)] = reader.strikes[k]
        print(np.isnan(reader.price['put'][k]).sum())

        tmp_corr = []
        for i in range(len(volatility[k])):
            tmp = [(x, y) for x, y in zip(reader.forward_average[k], volatility[k][i]) if not np.isnan(y)]
            tmp_spot = [tmp[i][0] for i in range(len(tmp))]
            tmp_volatility = [tmp[i][1] for i in range(len(tmp))]

            if len(tmp_volatility) != 0:
                tmp_corr.append(np.corrcoef(tmp_spot, tmp_volatility)[0, 1])
            else:
                tmp_corr.append(np.nan)
        tmp_corr = np.array(tmp_corr)

        df['Correlation coefficient ' + str(k)] = tmp_corr
        df.to_csv('forward_correlation' + reader.expiry_date[k].split()[0] + '.csv', index=False)

        tmp = tmp_corr[np.logical_not(np.isnan(tmp_corr))]
        indexes = np.arange(len(tmp_corr))
        indexes = indexes[np.logical_not(np.isnan(tmp_corr))]

        fig.append(make_subplots(rows=len(tmp), cols=1, start_cell="bottom-left",
                                 subplot_titles=[
                                     'r = ' + str(round(t, 3)) + ' for strike ' + str(reader.strikes[k][i])
                                     for i, t in zip(indexes, tmp)]))

        for i, index in zip(np.arange(len(indexes)), indexes):
            fig[k].add_trace(
                go.Scatter(x=reader.forward_average[k], y=volatility[k][index], mode="markers",  # reader.spot[k]
                           marker=dict(color=reader.time[k])), row=i + 1, col=1)

            # fig[k].add_trace(
            #     go.Scatter(x=reader.time[k], y=volatility[k][index], mode="lines",  # reader.spot[k]
            #                marker=dict(color=reader.time[k])), row=i + 1, col=1)

            fig[k].update_xaxes(title_text="Forward price", row=i + 1, col=1)
            fig[k].update_yaxes(title_text="Volatility", row=i + 1, col=1)

            fig[k].update_layout(showlegend=False)
            fig[k].update_layout(height=len(tmp) * 250, width=600,
                                 title_text="Correlation for expiration time " + str(
                                     reader.expiry_date[k]) + '\nfor ' + str(reader.price_type))

    for f in fig:
        f.show()



