import requests
import time
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.basemap import Basemap

plt.ion()  # interactive mode
m = Basemap(projection='moll', resolution=None, lat_0=0, lon_0=0)
m.bluemarble(scale=0.5)

plt.title("ISS position")
line, = m.plot([], [], 'ro')

xs = []
ys = []
while True:
    resp = requests.get("http://api.open-notify.org/iss-now.json")
    if resp.status_code != 200:
        time.sleep(10)
        continue

    lat = float(resp.json()['iss_position']['latitude'])
    lng = float(resp.json()['iss_position']['longitude'])
    x, y = m(lng, lat)
    ys.append(y)
    xs.append(x)

    line.set_xdata(xs)
    line.set_ydata(ys)
    plt.draw()  # force update
    plt.pause(1)  # wait
