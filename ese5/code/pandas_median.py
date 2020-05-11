import pandas as pd
import numpy as np

data = pd.read_csv('./aapl.us.txt')
values = data['Open']
above = data['Close'] > np.median(values)
data['above_median'] = above
data.to_csv('./result.csv')
