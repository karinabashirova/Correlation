import pandas as pd
from math import *
import numpy as np
import warnings
import os
from scipy import stats
import argparse
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


class Price:
    def __init__(self, type):
        if type == 'ask_c' or type == 'bid_c' or type == 'mid_c':
            self.type = 'call'
        else:
            self.type = 'put'


class Reader:
    def __init__(self, type, path):
        self.df_options = pd.DataFrame()
        self.path = path
        self.strikes = []
        self.price = []
        self.time = []
        self.spot = []
        self.v = []
        pr = Price(type)
        self.option_type = pr.type
        self.price_type = type
        self.expiry_date = []

    def get_data_from_file(self):
        for k, folder in enumerate(os.listdir(self.path)):
            check = False
            tmp_spot = []
            tmp_time = []
            price = []

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
                    price.append((np.array(df.loc[df['k'].isin(self.strikes[k])]['ask_' + self.option_type[0]].values.tolist()) + np.array(df.loc[df['k'].isin(self.strikes[k])]['bid_' + self.option_type[0]].values.tolist()))/2)
                else:
                    price.append(df.loc[df['k'].isin(self.strikes[k])][self.price_type].values.tolist())

            self.spot.append(tmp_spot)
            self.time.append(tmp_time)
            self.price.append(np.array(price).T)

    def count_vol(self):
        v = self.price.copy()
        for k in range(len(self.price)):
            for i in range(len(self.price[k])):
                vol = Volatility(self.option_type)
                print('.', end='')
                for j in range(len(self.price[k][i])):
                    if not np.isnan(self.price[k][i][j]):
                        v[k][i][j] = vol.sigma_from_price(self.price[k][i][j], self.strikes[k][i], self.time[k][j], 0.,
                                                          0.001, self.spot[k][j])
                    else:
                        v[k][i][j] = np.nan
        self.v = v


class Volatility:
    def __init__(self, option_type):
        self.option_type = option_type

    def sigma_from_price(self, C, strike, expiry, r, error, S):
        sigma = 1.
        dv = error + 1.
        count = 0
        while count <= 100:
            # while abs(dv) > error:
            count += 1
            d1 = (log(S / strike) + (r + 0.5 * sigma ** 2) * expiry) / (sigma * sqrt(expiry))
            d2 = d1 - sigma * sqrt(expiry)
            price = self.C_(d1, d2, strike, r, expiry, S)
            vega = self.Vega(S, d1, expiry)
            price_error = price - C
            dv = 1. * price_error / vega
            sigma = (sigma - dv)
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

    def Vega(self, S_, d1, expiry):
        vega = S_ * stats.norm.pdf(d1) * sqrt(expiry)
        return vega

    def C_(self, d1, d2, E, r, expiry, S):
        if self.option_type == 'call':
            # print(E < S, S * self.N(d1), E * exp(-r * expiry) * self.N(d2), S * self.N(d1) - E * exp(-r * expiry) * self.N(d2))
            return S * self.N(d1) - E * exp(-r * expiry) * self.N(d2)
        else:
            return -S * self.N(-d1) + E * exp(-r * expiry) * self.N(-d2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Correlation between volatility and underlying price')
    parser.add_argument("t", type=str, choices=['ask_c', 'bid_c', 'mid_c', 'ask_p', 'bid_p', 'mid_p'],
                        help="Surface for ask_c/bid_c/mid_c")
    parser.add_argument('p', help='Path to files')

    args = parser.parse_args()

    reader = Reader(args.t, args.p)
    reader.get_data_from_file()
    reader.count_vol()

    volatility = reader.v
    fig = []
    for k in range(len(volatility)):
        df = pd.DataFrame()
        df['Strike ' + str(k)] = reader.strikes[k]

        tmp_corr = []
        tmp_spot = []
        tmp_vol = []
        for i in range(1, len(reader.spot[k])):
            tmp_spot.append(reader.spot[k][i] - reader.spot[k][i-1])

        for i in range(1, len(volatility[k])):
            tmp_vol.append(np.array(volatility[k][i]) - np.array(volatility[k][i-1]))

        print(len(volatility[k]), len(tmp_spot), len(tmp_vol), len(tmp_vol[0]))

        for i in range(len(volatility[k]) - 1):
            tmp_corr.append(np.corrcoef(tmp_spot, tmp_vol[i][1:])[0, 1])  # reader.spot[k]

        # df['Correlation coefficient ' + str(k)] = tmp_corr
        # df.to_csv('correlation' + reader.expiry_date[k].split()[0] + '.csv', index=False)
        tmp_corr = np.array(tmp_corr)

        tmp = tmp_corr[np.logical_not(np.isnan(tmp_corr))]
        indexes = np.arange(len(tmp_corr))
        indexes = indexes[np.logical_not(np.isnan(tmp_corr))]

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=np.arange(len(reader.spot[0])), y=reader.spot[0], mode="markers+lines"))
        # fig1.add_trace(go.Scatter(x=[2]*len(reader.spot[1]), y=reader.spot[1], mode="markers"))
        fig1.show()

        try:
            if len(tmp) > 1:
                print('plt')
                fig.append(make_subplots(rows=len(tmp), cols=1, start_cell="bottom-left",
                                         subplot_titles=['r = ' + str(round(t, 3)) + ' for strike ' + str(reader.strikes[k][i]) for
                                             i, t in zip(indexes, tmp)]))
                for i, index in zip(np.arange(len(indexes)), indexes):
                    fig[k].add_trace(go.Scatter(x=tmp_spot, y=tmp_vol[index], mode="markers",
                                                name="Option with strike " + str(reader.strikes[k][i])), row=i + 1,
                                     col=1)
                    fig[k].update_xaxes(title_text="Spot", row=i + 1, col=1)
                    fig[k].update_yaxes(title_text="Volatility", row=i + 1, col=1)
                    fig[k].update_layout(showlegend=False)
                    fig[k].update_layout(height=len(tmp) * 250, width=600,
                                         title_text="Correlation for expiration time " + str(reader.expiry_date[k]) + '\nfor ' + str(reader.price_type))


            else:
                print('raise')
                raise IndexError
        except IndexError:
            print('except')
            if len(tmp) == 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=reader.spot[k], y=volatility[k][0], mode="markers",
                                         name="Option with strike " + str(reader.strikes[k][i])), row=i + 1, col=1)
                fig.update_xaxes(title_text="Spot", row=i + 1, col=1)
                fig.update_yaxes(title_text="Volatility", row=i + 1, col=1)
                fig.update_layout(showlegend=False)
                fig.update_layout(height=len(tmp) * 250, width=600,
                                  title_text="Correlation for expiration time " + str(reader.expiry_date[k]) + '\nfor ' + str(reader.price_type))
                fig.show()
            else:
                pass

    for f in fig:
        f.show()