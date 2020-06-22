import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MY_DATASET_PATH = '../../Stock market Data/Stock market data per company/'
SELL_RATIO = 1.005
BUY_RATIO = 0.995
K = 4

def load_dataset():
    dataset = {}
    file_list = os.listdir(MY_DATASET_PATH)
    for filename in file_list:
        if '.us.txt' in filename:
            company_key = filename.split('.')[0]
            try:
                df = pd.read_csv(MY_DATASET_PATH + filename)
                dataset[company_key] = df
            except:
                continue

    return dataset

def generate_roi_lists(dataset):
    roi_keep = []
    roi_gain = []
    for company in dataset:
        df = dataset[company]
        prices = df['Close'].values
        bk, sk = keep_strategy(prices)
        bg, sg = gain_strategy(prices, SELL_RATIO, BUY_RATIO)

        roi_keep.append(roi(prices, bk, sk))
        roi_gain.append(roi(prices, bg, sg))
    return roi_keep, roi_gain

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
        sells.append(-1)
    gains = prices[sells] / prices[buys]
    gains -= 1
    return np.prod(gains)

def k_means_init(domain, k):
    centroids = {}
    for i in range(k):
        point = []
        for dim in domain:
            point.append(np.random.uniform(dim[0], dim[1]))
        centroids[i] = point
    return centroids

def k_means_assign(points, centroids):
    assignments = []
    for point in points:
        p = np.array(point)
        distances = [np.linalg.norm(p - centroids[i]) for i in
                     sorted(centroids)]
        assignments.append(np.argmin(distances))
    return assignments

def k_means_update(points, assignments, old_centroids):
    new_centroids = {}
    k = max(assignments) + 1
    assignments = np.array(assignments)
    for i in range(k):
        assigned_points = points[assignments == i]
        if len(assigned_points) == 0:
            new_centroids[i] = old_centroids[i]
        else:
            new_centroids[i] = np.mean(assigned_points, axis=0)
    return new_centroids



def plot_centroids_and_assignments(centroids, assignments, xs, ys, colmap):
    # centroids + colored scatter plot
    plt.xlabel('keep')
    plt.ylabel('gain')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i])
    colors = [colmap[a] for a in assignments]
    plt.scatter(xs, ys, color=colors, alpha=0.2)
    plt.show()

def plot_assignments_and_move_centroids(xs, ys, colmap, old_centroids,
                                        new_centroids, assignments):
    # centroids + colored scatter plot
    plt.xlabel('keep')
    plt.ylabel('gain')
    for i in new_centroids.keys():
        plt.scatter(*new_centroids[i], color=colmap[i])
    colors = [colmap[a] for a in assignments]
    plt.scatter(xs, ys, color=colors, alpha=0.2)
    #arrows
    for i in new_centroids:
        old_x = old_centroids[i][0]
        old_y = old_centroids[i][1]
        dx = new_centroids[i][0] - old_x
        dy = new_centroids[i][1] - old_y
        plt.arrow(old_x, old_y, dx, dy, color=colmap[i])
    plt.show()


dataset = load_dataset()
print(len(dataset))
roi_keep, roi_gain = generate_roi_lists(dataset)

# scatter plot
plt.scatter(roi_keep, roi_gain, color='k')
plt.xlabel('Keep')
plt.ylabel('Gain')
plt.show()

# DOMAIN COMPUTATION
domain = [[min(dimension), max(dimension)] for dimension in [roi_keep,
                                                             roi_gain]]
print(domain)
centroids = k_means_init(domain, K)
# scatter plot
plt.scatter(roi_keep, roi_gain, color='k')
plt.xlabel('Keep')
plt.ylabel('Gain')
colmap = {0: 'y', 1: 'r', 2: 'g', 3: 'b'}
for i in centroids:
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()

points = [(x, y) for x, y in zip(roi_keep, roi_gain)]
assignments = k_means_assign(points, centroids)
plot_centroids_and_assignments(centroids, assignments, roi_keep, roi_gain,
                               colmap)

new_centroids = k_means_update(points, assignments, centroids)
plot_assignments_and_move_centroids(roi_keep, roi_gain, colmap, centroids,
                                    new_centroids, assignments)

change = any([any(new_centroids[i] != centroids[i]) for i in range(K)])
while change:
    centroids = new_centroids
    assignments = k_means_assign(points, centroids)
    plot_centroids_and_assignments(centroids, assignments, roi_keep, roi_gain,
                                   colmap)

    new_centroids = k_means_update(points, assignments, centroids)
    plot_assignments_and_move_centroids(roi_keep, roi_gain, colmap, centroids,
                                        new_centroids, assignments)
    change = any([any(new_centroids[i] != centroids[i]) for i in range(K)])



plot_centroids_and_assignments(centroids, assignments, roi_keep, roi_gain,
                               colmap)
