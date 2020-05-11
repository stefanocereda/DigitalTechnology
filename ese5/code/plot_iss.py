import requests
import time
import matplotlib.pyplot as plt

plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title("ISS position")
line, = plt.plot([], [], 'bo')
plt.ion()  # interactive mode

latitudes = []
longitudes = []
while True:
    resp = requests.get("http://api.open-notify.org/iss-now.json")
    if resp.status_code != 200:
        time.sleep(10)
        continue

    latitudes.append(float(resp.json()['iss_position']['latitude']))
    longitudes.append(float(resp.json()['iss_position']['longitude']))

    line.set_xdata(longitudes)
    line.set_ydata(latitudes)
    plt.draw()  # force update
    plt.pause(1)  # wait
