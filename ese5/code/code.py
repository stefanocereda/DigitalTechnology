import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./result.csv')
above = data['above_median']

plt.plot(data['Open'], 'r+')
plt.show()
