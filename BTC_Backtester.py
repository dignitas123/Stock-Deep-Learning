import re
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta


class _Backtest:
    def __init__(self, df, h2l_std,
                 commission=1.2e-4,
                 tp_multiplier=3,
                 stop_multiplier=2,
                 trailing=False,
                 trail_period=20):

        self.trailing = trailing
        self.trailing_period = trail_period
        self.commission = commission
        self._h2l_std = h2l_std
        self.tp_mult = tp_multiplier
        self.stop_mult = stop_multiplier

        self._data = df.to_numpy(dtype=np.float64).T
        self._data_len = self._data.shape[1]
        self._H = self._data[1]
        self._L = self._data[2]
        self._C = self._data[3]
        self.EMA200 = ta.EMA(self._C, timeperiod=200)
        self.EMA20 = ta.EMA(self._C, timeperiod=20)
        self.trailEMA = ta.EMA(self._C, timeperiod=self.trailing_period)
        self.dates = list(df.index.values)
        self.hours = [parse_datestring_2(
            self.dates[t])[0] for t in range(self._data_len)]
        self.minutes = [parse_datestring_2(
            self.dates[t])[1] for t in range(self._data_len)]
        self._all_sl = self.all_SL()
        # print("all sl calculated")

        """
        self._close_dists = lambda i: [
            (self._C[i-x]-self._C[i])/self._all_sl[i]
            if i > 10 else 0 for x in range(10)]
        self._all_close_dists = [
            self._close_dists(j) for j in range(self._data_len)]
        # print("all close dists calculated")
        self._high_dists = lambda i: [
            (self._H[i-x]-self._H[i])/self._all_sl[i]
            if i > 2 else 0 for x in range(2)]
        self._all_high_dists = [
            self._high_dists(j) for j in range(self._data_len)]
        # print("all high dists calculated")
        self._low_dists = lambda i: [
            (self._L[i-x]-self._L[i])/self._all_sl[i]
            if i > 2 else 0 for x in range(2)]
        self._all_low_dists = [
            self._low_dists(j) for j in range(self._data_len)]
        """
        # print("all low dists calculated")
        self.reset()

    def reset(self):
        # interval timestep from one interval in idx_pairs to the next one
        self.t = 0
        self.sl_distance = 0  # stoploss distance
        self.pnl_r = []  # profit/losses tracking in R's
        self.close_times = []  # close times in date format
        self.positions = []  # stores entry price of a position
        self.isLong = True
        self.position_sl = 0
        self.position_tp = 0
        self.open_pos = 0

        """
        position value in points
        (example: _entry 1.5 _current close 2 = 0.5 if long)
        """
        self.position_value = 0

    def true_range(self, i):
        return max(self._H[i]-self._L[i],
                   abs(self._H[i]-self._C[i-1]),
                   abs(self._L[i]-self._C[i-1]))

    def all_SL(self):
        # standard deviation of h2l/c moving average
        tr, sl = [np.nan], [self._C[0] * self._h2l_std * 4]
        for i in range(1, self._data_len):
            tr.append(self.true_range(i))
            if len(tr) == 501:
                sl.append(np.mean(tr[-500:]) * 2)
                tr.pop(0)
            else:
                sl.append(self._C[i] * self._h2l_std * 4)

        return sl

    def go_short(self):
        self.positions.append(self._C[self.t])
        self.set_sl_tp_short()
        self.isLong = False
        self.open_pos = len(self.positions)

    def go_long(self):
        self.positions.append(self._C[self.t])
        self.set_sl_tp_long()
        self.isLong = True
        self.open_pos = len(self.positions)

    def set_sl_tp_long(self):
        entry = self.entry_price()
        self.position_sl = entry - self._all_sl[self.t] * self.stop_mult
        self.sl_distance = entry - self.position_sl
        self.position_tp = entry + self.sl_distance * self.tp_mult

    def set_sl_tp_short(self):
        entry = self.entry_price()
        self.position_sl = entry + self._all_sl[self.t] * self.stop_mult
        self.sl_distance = self.position_sl - entry
        self.position_tp = entry - self.sl_distance * self.tp_mult

    def sl_is_hit_short(self):
        return self._H[self.t] >= self.position_sl

    def sl_is_hit_long(self):
        return self._L[self.t] <= self.position_sl

    def tp_is_hit_long(self):
        if self.trailing:
            return (
                self._C[self.t] < self._C[self.t-1] and self._C[self.t] < self._L[self.t-1] and self._C[self.t-1] < self._C[self.t-2] and self._C[self.t-2] < self._C[self.t-3])
        else:
            return self._H[self.t] > self.position_tp

    def tp_is_hit_short(self):
        if self.trailing:
            return (
                self._C[self.t] > self._C[self.t-1] and self._C[self.t] > self._H[self.t-1] and self._C[self.t-1] > self._C[self.t-2] and self._C[self.t-2] > self._C[self.t-3])
        else:
            return self._L[self.t] < self.position_tp

    def entry_price(self):
        return np.mean(self.positions)

    def register_closed_pos(self, reward_d, state):
        """
        reward_d: reward distance in points
        """
        # calculate costs of closing this position, which is 2x the
        # commission (in percentage) and 2x - a full "roundturn"
        cost = self._C[self.t] * self.commission * 2

        # the reward/risk of the closed position
        rr = round(((reward_d - cost) / self.sl_distance) * 100)

        # add profits in R to pnl_r list
        self.pnl_r.append(rr/100)
        self.close_times.append(self.dates[self.t])
        # no open position anymore
        self.positions = []
        self.open_pos = 0

        return rr

    def step(self):
        if self.open_pos > 0:
            if self.isLong:
                if self.sl_is_hit_long():
                    self.register_closed_pos(-self.sl_distance, "stopped out")
                elif self.tp_is_hit_long():
                    if self.trailing:
                        self.register_closed_pos(self._C[self.t] - self.entry_price(), "tp is hit")
                    else:
                        self.register_closed_pos(self.sl_distance * self.tp_mult, "tp is hit")
            else:
                if self.sl_is_hit_short():
                    self.register_closed_pos(-self.sl_distance, "stopped out")
                elif self.tp_is_hit_short():
                    if self.trailing:
                        self.register_closed_pos(self.entry_price() - self._C[self.t], "tp is hit")
                    else:
                        self.register_closed_pos(self.sl_distance * self.tp_mult, "tp is hit")
        else:
            ema_condition_1 = self.EMA200[self.t] > self.EMA200[self.t-1]
            ema_condition_2 = self.EMA20[self.t] > self.EMA20[self.t-1]

            if self.t > 1 and ema_condition_1 and ema_condition_2:
                if self._C[self.t] > self._H[self.t-1]:
                    self.go_long()
            else:
                ema_condition_1 = self.EMA200[self.t] < self.EMA200[self.t-1]
                ema_condition_2 = self.EMA20[self.t] < self.EMA20[self.t-1]
                if ema_condition_1 and ema_condition_2:
                    if self._C[self.t] < self._L[self.t-1]:
                        self.go_short()

        self.t += 1


def makeNewList(my_list):
    # for making plots of list values, added to account balance values
    new_list = []
    sumList = 0
    for el in my_list:
        sumList += el
        new_list.append(sumList)

    return new_list


def range_list(stop, start=1, step=1):
    # returns a list of numbers with inputs until when 'stop', 'start' and 'step'
    i = start
    res = []
    while i < stop + step:
        res.append(i)
        i += step
    return res


def parse_datestring(s):
    # s = 29/05/2000 02:05:00
    p = re.compile(r"(\d{2})[-/]\d{2}[-/](\d{4}) \d{2}:\d{2}:\d{2}")
    return np.array(p.match(s).groups(), dtype=int)  # day, month


def parse_datestring_2(s):
    # s = 29/05/2000 02:05:00
    p = re.compile(r"\d{2}[-/]\d{2}[-/]\d{4} (\d{2}):(\d{2}):\d{2}")
    return np.array(p.match(s).groups(), dtype=int)  # hour, minute


def plot_pnl_stats(env):
    env.reset()

    for _ in range(env._data_len):
        env.step()

    # stat calculations:
    _equity = [0, ]
    profits = []
    losses = []
    trade_count = 0  # trades not 0
    highest = 0
    highest_rel_dd = 0
    for r_pnl in env.pnl_r:
        _equity.append(_equity[-1] + r_pnl)
        if r_pnl > 0:
            profits.append(r_pnl)
            trade_count += 1
        else:
            losses.append(r_pnl)
            trade_count += 1
        if _equity[-1] > highest:
            highest = _equity[-1]

        if _equity[-1] < highest:
            dd = highest - _equity[-1]
            if dd > highest_rel_dd:
                highest_rel_dd = dd

    steps = range_list(len(_equity))
    plt.plot(steps, _equity)

    plt.xlabel('Trades')
    plt.ylabel('R Multiples')
    plt.title('Equity')
    plt.show()

    # print out stats
    hr, avgp, avgl = len(profits)/trade_count, np.mean(profits), abs(np.mean(losses))
    total_profit = sum(env.pnl_r)
    # print(_env.pnl_r) # optional: can print out all pnl_r values in a list
    print("Stats\n########################\n")
    print('R Profits:', round(total_profit, 2))
    print('Trades:', len(profits)+len(losses))
    print('Hitrate:', round(hr, 2))
    print('Average Loss:', round(avgl, 2))
    print('Average Risk/Reward:', round(avgp/avgl, 2))
    print('Profit Factor:', round((avgp*hr)/(avgl*(1-hr)), 2))
    print('Biggest Rel. DD:', round(highest_rel_dd, 2),)
    print('Gain/DD Ratio:', round(total_profit/highest_rel_dd, 2), '\n')
    # print(list(zip(env.pnl_r, env.close_times)))


# DATA PREPARATION
# ################

# Iota
# main_name = "BTC_15m_clean_new_30m"

# Lotus
main_name = "BTC_15m_clean_new3"

# main_name = "BTC_15m_clean_30m"
data = pd.read_pickle(main_name)
# h2l_std = 0.0081
h2l_std = 0.0088
# h2l_std = 0.0126
print("Data rows:", len(data))
print(data.head())


# ENVIRONMENT CREATION
# ####################

# Iota
# env = _Backtest(data, h2l_std, tp_multiplier=2.75, stop_multiplier=1,
                # trailing=True, trail_period=25)

# Lotus
env = _Backtest(data, h2l_std, tp_multiplier=3, stop_multiplier=1,
                trailing=False, trail_period=25)

plot_pnl_stats(env)
