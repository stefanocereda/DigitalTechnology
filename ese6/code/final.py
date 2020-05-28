import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This works on my machine, you have to change it
MY_DATASET_PATH = '../../Stock market Data/Stock market data per company/'
# Values fo gain strategy
SELL_RATIO = 1.005
BUY_RATIO = 0.995
# Number of k-means centroids
K = 4

def load_dataset():
    dataset = {}
    file_list = os.listdir(MY_DATASET_PATH)
    for filename in file_list:
        # check that this is a valid file
        if '.us.txt' in filename:
            # get the characters before the first dot
            company_name = filename.split('.')[0]
            # load the csv building the correct path
            try:
                df = pd.read_csv(MY_DATASET_PATH + filename)
            except:
                continue
            # save in the dictionary
            dataset[company_name] = df
    return dataset

def generate_roi_lists(dataset):
    roi_keep = []
    roi_gain = []
    for company in dataset:
        df = dataset[company]
        # run the strategies
        prices = df['Close'].values
        bk, sk = keep_strategy(prices)
        bg, sg = gain_strategy(prices, sell_ratio=SELL_RATIO,
                               buy_ratio=BUY_RATIO)
        # compute and save the ROIs
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
    # This has to work with any number of dimensions, even if our example is 2D
    centroids = {}
    for i in range(k):
        point = []
        for d in domain:
            point.append(np.random.uniform(d[0], d[1]))
        centroids[i] = np.array(point)
    return centroids

def k_means_assignment(points, centroids):
    assignments = []
    for point in points:
        p = np.array(point)
        distances = [np.linalg.norm(p - centroids[i]) for i in
                     sorted(centroids)]
        assignments.append(np.argmin(distances))
    return assignments

def k_means_update(points, assignments, old_centroids):
    k = max(assignments) + 1
    new_centroids = {}
    assignments = np.array(assignments)
    points = np.array(points)
    for i in range(k):
        assigned_points = points[assignments == i]
        if len(assigned_points) == 0:
            new_centroids[i] = old_centroids[i]
        else:
            new_centroids[i] = np.mean(assigned_points, axis = 0)
    return new_centroids

def plot_centroids_and_assignments(centroids, xs, ys, colmap):
    # centroids + colored scatter plot
    plt.xlabel('keep')
    plt.ylabel('gain')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i])
    colors = [colmap[a] for a in assignments]
    plt.scatter(xs, ys, color=colors, alpha=0.2)
    plt.show()

def plot_assignements_and_move_centroids(xs, ys, colmap, centroids,
                                         new_centroids):
    plt.xlabel('keep')
    plt.ylabel('gain')
    for i in new_centroids.keys():
        plt.scatter(*new_centroids[i], color=colmap[i])
    colors = [colmap[a] for a in assignments]
    plt.scatter(xs, ys, color=colors, alpha=0.2)
    # arrow for moving the centroids
    for i in centroids.keys():
        old_x = centroids[i][0]
        old_y = centroids[i][1]
        try:
            dx = new_centroids[i][0] - old_x
        except:
            breakpoint()
        dy = new_centroids[i][1] - old_y
        plt.arrow(old_x, old_y, dx, dy, color=colmap[i])
    plt.show()


dataset = load_dataset()
roi_keep, roi_gain = generate_roi_lists(dataset)

# scatter plot
plt.scatter(roi_keep, roi_gain, color='k')
plt.xlabel('keep')
plt.ylabel('gain')
plt.show()

# where do our points live?
domain = [[min(r), max(r)] for r in [roi_keep, roi_gain]]
print(domain)
centroids = k_means_init(domain, K)
# scatter plot + centroids
plt.scatter(roi_keep, roi_gain, color='k')
plt.xlabel('keep')
plt.ylabel('gain')
colmap = {0: 'y', 1: 'r', 2: 'g', 3: 'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()

points = [(x, y) for x, y in zip(roi_keep, roi_gain)]
assignments = k_means_assignment(points, centroids)
plot_centroids_and_assignments(centroids, roi_keep, roi_gain, colmap)

new_centroids = k_means_update(points, assignments, centroids)
plot_assignements_and_move_centroids(roi_keep, roi_gain, colmap, centroids,
                                     new_centroids)

# Loop
change = any([any(new_centroids[i] != centroids[i]) for i in range(K)])
while change:
    centroids = new_centroids
    assignments = k_means_assignment(points, centroids)
    plot_centroids_and_assignments(centroids, roi_keep, roi_gain, colmap)

    new_centroids = k_means_update(points, assignments, centroids)
    plot_assignements_and_move_centroids(roi_keep, roi_gain, colmap, centroids,
    new_centroids)

    change = any([any(new_centroids[i] != centroids[i]) for i in range(K)])

# Plot the final assignements
plt.title('Final assignement')
plot_centroids_and_assignments(centroids, roi_keep, roi_gain, colmap)
