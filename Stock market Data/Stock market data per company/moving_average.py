import pandas as pd
import numpy as np
import time

tick = 'wafd'  # Washington Federal, Inc.
suffix = '.us.txt'
stock = pd.read_csv('./' + tick + suffix)

def moving_average_l(signal, period):
    result = []
    for day in range(period-1):
        result.append(np.nan)
    for end_day in range(period-1, len(signal)):
        start_day = end_day - period + 1
        values = signal[start_day: end_day+1]
        result.append(np.average(values))
    return np.array(result)

def moving_average_a(signal, period):
    cumsum = np.cumsum(signal)
    sum_start = np.concatenate(([0], cumsum[:-period]))
    sum_end = cumsum[period-1:]
    result = (sum_end - sum_start) / period
    fill = np.full(period-1, np.nan)
    return np.concatenate((fill, result))


print(stock.head)
tic = time.time()
ma10_l = moving_average_l(stock['Close'].values, 10)
toc = time.time()
tl = toc-tic
print("List version took {} seconds".format(toc-tic))

tic = time.time()
ma10_a = moving_average_a(stock['Close'].values, 10)
toc = time.time()
ta = toc-tic
print("Array version took {} seconds".format(toc-tic))
print("We would save {} seconds".format((tl-ta) * 7196))


def keep_strategy(prices):
    return [0], [len(prices) - 1]  # buy and sell days

def gain_strategy(prices, sell_ratio, buy_ratio):
    buys = [0]
    sells = []
    action_price = prices[0] * sell_ratio

    for day, price in enumerate(prices):
        # do we have any stock?
        if len(sells) < len(buys):
            # should we sell?
            if price >= action_price:
                sells.append(day)
                action_price = price * buy_ratio
        # no stock, should we buy?
        elif price <= action_price:
            buys.append(day)
            action_price = price * sell_ratio
    return buys, sells


def roi(prices, buys, sells):
    # simulate selling last day
    if len(sells) < len(buys):
        # np.append works like list.append
        np.append(sells, -1)
    gains = prices[sells] / prices[buys]
    gains -= 1
    return np.prod(gains)

prices = stock['Close']
bk, sk = keep_strategy(prices).values
bg, sg = gain_strategy(prices, sell_ratio=1.1, buy_ratio=0.9)

print("Simple strategy ROI: {}".format(roi(prices, bk, sk)))
print("Complex strategy ROI: {}".format(roi(prices, bg, sg)))
