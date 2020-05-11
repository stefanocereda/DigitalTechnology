import matplotlib.pyplot as plt
xs = range(10,100,2)
ys = [x ** 2 for x in xs]
plt.plot(xs, ys)
plt.xlabel("Even numbers")
plt.ylabel("Squared numbers")
plt.show()
