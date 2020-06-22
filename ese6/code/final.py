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

# Function for exercise 1
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

# Function for exercise 2
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

# This comes from previous session
def keep_strategy(prices):
    return [0], [len(prices) - 1]  # buy and sell days

# This comes from previous session
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

# This comes from previous session
def roi(prices, buys, sells):
    # simulate selling last day
    if len(sells) < len(buys):
        sells.append(-1)
    gains = prices[sells] / prices[buys]
    gains -= 1
    return np.prod(gains)

# Exercise 4: kmeans init
def k_means_init(domain, k):
    # This has to work with any number of dimensions, even if our example is 2D
    centroids = {}
    for i in range(k):
        point = []
        for d in domain:
            point.append(np.random.uniform(d[0], d[1]))
        centroids[i] = np.array(point)
    return centroids

# Exercise 5: kmeans assignment
def k_means_assignment(points, centroids):
    assignments = []
    for point in points:
        p = np.array(point)
        distances = [np.linalg.norm(p - centroids[i]) for i in
                     sorted(centroids)]
        assignments.append(np.argmin(distances))
    return assignments

# Exercise 6 kmeans update
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

# Plot function used in exercise 5
def plot_centroids_and_assignments(centroids, assignments, xs, ys, colmap):
    # centroids + colored scatter plot
    plt.xlabel('keep')
    plt.ylabel('gain')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i])
    colors = [colmap[a] for a in assignments]
    plt.scatter(xs, ys, color=colors, alpha=0.2)
    plt.show()

# Plot function used in exercise 6
def plot_assignements_and_move_centroids(xs, ys, colmap, centroids,
                                         new_centroids, assignments):
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
        dx = new_centroids[i][0] - old_x
        dy = new_centroids[i][1] - old_y
        plt.arrow(old_x, old_y, dx, dy, color=colmap[i])
    plt.show()


# This if is optional
if __name__ == '__main__':
    # Exercise 1
    dataset = load_dataset()
    # Exercise 2
    roi_keep, roi_gain = generate_roi_lists(dataset)

    # Exercise 3: scatter plot
    plt.scatter(roi_keep, roi_gain, color='k')
    plt.xlabel('keep')
    plt.ylabel('gain')
    plt.show()

    # Exercise 4: kmeans init
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

    # Zip function for the list of points
    points = [(x, y) for x, y in zip(roi_keep, roi_gain)]
    # Exercise 5: kmeans assignment
    assignments = k_means_assignment(points, centroids)
    plot_centroids_and_assignments(centroids, assignments, roi_keep, roi_gain, colmap)

    # Exercise 6: kmeans update
    new_centroids = k_means_update(points, assignments, centroids)
    plot_assignements_and_move_centroids(roi_keep, roi_gain, colmap, centroids,
                                         new_centroids, assignments)

    # Exercise 7: loop until nothing changes
    change = any([any(new_centroids[i] != centroids[i]) for i in range(K)])
    while change:
        centroids = new_centroids
        assignments = k_means_assignment(points, centroids)
        plot_centroids_and_assignments(centroids, assignments, roi_keep,
                                       roi_gain, colmap)

        new_centroids = k_means_update(points, assignments, centroids)
        plot_assignements_and_move_centroids(roi_keep, roi_gain, colmap, centroids,
                                             new_centroids, assignments)

        change = any([any(new_centroids[i] != centroids[i]) for i in range(K)])

    # Plot the final assignements
    plt.title('Final assignement')
    plot_centroids_and_assignments(centroids, assignments, roi_keep, roi_gain, colmap)
