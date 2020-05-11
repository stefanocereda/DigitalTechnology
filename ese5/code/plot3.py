import matplotlib.pyplot as plt
import numpy as np
xs = np.arange(0, 2*np.pi, 0.5)
ys_c = np.cos(xs)
ys_s = np.sin(xs)
plt.plot(xs, ys_c, 'r--')
plt.plot(xs, ys_s, 'bo')
plt.show()
